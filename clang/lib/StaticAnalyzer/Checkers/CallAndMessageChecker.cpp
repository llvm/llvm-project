//===--- CallAndMessageChecker.cpp ------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines CallAndMessageChecker, a builtin checker that checks for various
// errors of call and objc message expressions.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/AST/ParentMap.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallString.h"

using namespace clang;
using namespace ento;

namespace {
class CallAndMessageChecker
  : public Checker< check::PreStmt<CallExpr>, check::PreObjCMessage,
                    check::PreCall > {
  mutable OwningPtr<BugType> BT_call_null;
  mutable OwningPtr<BugType> BT_call_undef;
  mutable OwningPtr<BugType> BT_cxx_call_null;
  mutable OwningPtr<BugType> BT_cxx_call_undef;
  mutable OwningPtr<BugType> BT_call_arg;
  mutable OwningPtr<BugType> BT_msg_undef;
  mutable OwningPtr<BugType> BT_objc_prop_undef;
  mutable OwningPtr<BugType> BT_objc_subscript_undef;
  mutable OwningPtr<BugType> BT_msg_arg;
  mutable OwningPtr<BugType> BT_msg_ret;
public:

  void checkPreStmt(const CallExpr *CE, CheckerContext &C) const;
  void checkPreObjCMessage(const ObjCMethodCall &msg, CheckerContext &C) const;
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;

private:
  static bool PreVisitProcessArg(CheckerContext &C, SVal V,
                                 SourceRange argRange, const Expr *argEx,
                                 bool IsFirstArgument, bool checkUninitFields,
                                 const CallEvent &Call, OwningPtr<BugType> &BT);

  static void emitBadCall(BugType *BT, CheckerContext &C, const Expr *BadE);
  void emitNilReceiverBug(CheckerContext &C, const ObjCMethodCall &msg,
                          ExplodedNode *N) const;

  void HandleNilReceiver(CheckerContext &C,
                         ProgramStateRef state,
                         const ObjCMethodCall &msg) const;

  static void LazyInit_BT(const char *desc, OwningPtr<BugType> &BT) {
    if (!BT)
      BT.reset(new BuiltinBug(desc));
  }
};
} // end anonymous namespace

void CallAndMessageChecker::emitBadCall(BugType *BT, CheckerContext &C,
                                        const Expr *BadE) {
  ExplodedNode *N = C.generateSink();
  if (!N)
    return;

  BugReport *R = new BugReport(*BT, BT->getName(), N);
  if (BadE) {
    R->addRange(BadE->getSourceRange());
    bugreporter::trackNullOrUndefValue(N, BadE, *R);
  }
  C.emitReport(R);
}

static StringRef describeUninitializedArgumentInCall(const CallEvent &Call,
                                                     bool IsFirstArgument) {
  switch (Call.getKind()) {
  case CE_ObjCMessage: {
    const ObjCMethodCall &Msg = cast<ObjCMethodCall>(Call);
    switch (Msg.getMessageKind()) {
    case OCM_Message:
      return "Argument in message expression is an uninitialized value";
    case OCM_PropertyAccess:
      assert(Msg.isSetter() && "Getters have no args");
      return "Argument for property setter is an uninitialized value";
    case OCM_Subscript:
      if (Msg.isSetter() && IsFirstArgument)
        return "Argument for subscript setter is an uninitialized value";
      return "Subscript index is an uninitialized value";
    }
    llvm_unreachable("Unknown message kind.");
  }
  case CE_Block:
    return "Block call argument is an uninitialized value";
  default:
    return "Function call argument is an uninitialized value";
  }
}

bool CallAndMessageChecker::PreVisitProcessArg(CheckerContext &C,
                                               SVal V, SourceRange argRange,
                                               const Expr *argEx,
                                               bool IsFirstArgument,
                                               bool checkUninitFields,
                                               const CallEvent &Call,
                                               OwningPtr<BugType> &BT) {
  if (V.isUndef()) {
    if (ExplodedNode *N = C.generateSink()) {
      LazyInit_BT("Uninitialized argument value", BT);

      // Generate a report for this bug.
      StringRef Desc = describeUninitializedArgumentInCall(Call,
                                                           IsFirstArgument);
      BugReport *R = new BugReport(*BT, Desc, N);
      R->addRange(argRange);
      if (argEx)
        bugreporter::trackNullOrUndefValue(N, argEx, *R);
      C.emitReport(R);
    }
    return true;
  }

  if (!checkUninitFields)
    return false;
  
  if (const nonloc::LazyCompoundVal *LV =
        dyn_cast<nonloc::LazyCompoundVal>(&V)) {

    class FindUninitializedField {
    public:
      SmallVector<const FieldDecl *, 10> FieldChain;
    private:
      StoreManager &StoreMgr;
      MemRegionManager &MrMgr;
      Store store;
    public:
      FindUninitializedField(StoreManager &storeMgr,
                             MemRegionManager &mrMgr, Store s)
      : StoreMgr(storeMgr), MrMgr(mrMgr), store(s) {}

      bool Find(const TypedValueRegion *R) {
        QualType T = R->getValueType();
        if (const RecordType *RT = T->getAsStructureType()) {
          const RecordDecl *RD = RT->getDecl()->getDefinition();
          assert(RD && "Referred record has no definition");
          for (RecordDecl::field_iterator I =
               RD->field_begin(), E = RD->field_end(); I!=E; ++I) {
            const FieldRegion *FR = MrMgr.getFieldRegion(*I, R);
            FieldChain.push_back(*I);
            T = I->getType();
            if (T->getAsStructureType()) {
              if (Find(FR))
                return true;
            }
            else {
              const SVal &V = StoreMgr.getBinding(store, loc::MemRegionVal(FR));
              if (V.isUndef())
                return true;
            }
            FieldChain.pop_back();
          }
        }

        return false;
      }
    };

    const LazyCompoundValData *D = LV->getCVData();
    FindUninitializedField F(C.getState()->getStateManager().getStoreManager(),
                             C.getSValBuilder().getRegionManager(),
                             D->getStore());

    if (F.Find(D->getRegion())) {
      if (ExplodedNode *N = C.generateSink()) {
        LazyInit_BT("Uninitialized argument value", BT);
        SmallString<512> Str;
        llvm::raw_svector_ostream os(Str);
        os << "Passed-by-value struct argument contains uninitialized data";

        if (F.FieldChain.size() == 1)
          os << " (e.g., field: '" << *F.FieldChain[0] << "')";
        else {
          os << " (e.g., via the field chain: '";
          bool first = true;
          for (SmallVectorImpl<const FieldDecl *>::iterator
               DI = F.FieldChain.begin(), DE = F.FieldChain.end(); DI!=DE;++DI){
            if (first)
              first = false;
            else
              os << '.';
            os << **DI;
          }
          os << "')";
        }

        // Generate a report for this bug.
        BugReport *R = new BugReport(*BT, os.str(), N);
        R->addRange(argRange);

        // FIXME: enhance track back for uninitialized value for arbitrary
        // memregions
        C.emitReport(R);
      }
      return true;
    }
  }

  return false;
}

void CallAndMessageChecker::checkPreStmt(const CallExpr *CE,
                                         CheckerContext &C) const{

  const Expr *Callee = CE->getCallee()->IgnoreParens();
  ProgramStateRef State = C.getState();
  const LocationContext *LCtx = C.getLocationContext();
  SVal L = State->getSVal(Callee, LCtx);

  if (L.isUndef()) {
    if (!BT_call_undef)
      BT_call_undef.reset(new BuiltinBug("Called function pointer is an "
                                         "uninitalized pointer value"));
    emitBadCall(BT_call_undef.get(), C, Callee);
    return;
  }

  ProgramStateRef StNonNull, StNull;
  llvm::tie(StNonNull, StNull) = State->assume(cast<DefinedOrUnknownSVal>(L));

  if (StNull && !StNonNull) {
    if (!BT_call_null)
      BT_call_null.reset(
        new BuiltinBug("Called function pointer is null (null dereference)"));
    emitBadCall(BT_call_null.get(), C, Callee);
  }

  C.addTransition(StNonNull);
}

void CallAndMessageChecker::checkPreCall(const CallEvent &Call,
                                         CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  // If this is a call to a C++ method, check if the callee is null or
  // undefined.
  if (const CXXInstanceCall *CC = dyn_cast<CXXInstanceCall>(&Call)) {
    SVal V = CC->getCXXThisVal();
    if (V.isUndef()) {
      if (!BT_cxx_call_undef)
        BT_cxx_call_undef.reset(new BuiltinBug("Called C++ object pointer is "
                                               "uninitialized"));
      emitBadCall(BT_cxx_call_undef.get(), C, CC->getCXXThisExpr());
      return;
    }

    ProgramStateRef StNonNull, StNull;
    llvm::tie(StNonNull, StNull) = State->assume(cast<DefinedOrUnknownSVal>(V));

    if (StNull && !StNonNull) {
      if (!BT_cxx_call_null)
        BT_cxx_call_null.reset(new BuiltinBug("Called C++ object pointer "
                                              "is null"));
      emitBadCall(BT_cxx_call_null.get(), C, CC->getCXXThisExpr());
      return;
    }

    State = StNonNull;
  }

  // Don't check for uninitialized field values in arguments if the
  // caller has a body that is available and we have the chance to inline it.
  // This is a hack, but is a reasonable compromise betweens sometimes warning
  // and sometimes not depending on if we decide to inline a function.
  const Decl *D = Call.getDecl();
  const bool checkUninitFields =
    !(C.getAnalysisManager().shouldInlineCall() && (D && D->getBody()));

  OwningPtr<BugType> *BT;
  if (isa<ObjCMethodCall>(Call))
    BT = &BT_msg_arg;
  else
    BT = &BT_call_arg;

  for (unsigned i = 0, e = Call.getNumArgs(); i != e; ++i)
    if (PreVisitProcessArg(C, Call.getArgSVal(i), Call.getArgSourceRange(i),
                           Call.getArgExpr(i), /*IsFirstArgument=*/i == 0,
                           checkUninitFields, Call, *BT))
      return;

  // If we make it here, record our assumptions about the callee.
  C.addTransition(State);
}

void CallAndMessageChecker::checkPreObjCMessage(const ObjCMethodCall &msg,
                                                CheckerContext &C) const {
  SVal recVal = msg.getReceiverSVal();
  if (recVal.isUndef()) {
    if (ExplodedNode *N = C.generateSink()) {
      BugType *BT = 0;
      switch (msg.getMessageKind()) {
      case OCM_Message:
        if (!BT_msg_undef)
          BT_msg_undef.reset(new BuiltinBug("Receiver in message expression "
                                            "is an uninitialized value"));
        BT = BT_msg_undef.get();
        break;
      case OCM_PropertyAccess:
        if (!BT_objc_prop_undef)
          BT_objc_prop_undef.reset(new BuiltinBug("Property access on an "
                                                  "uninitialized object "
                                                  "pointer"));
        BT = BT_objc_prop_undef.get();
        break;
      case OCM_Subscript:
        if (!BT_objc_subscript_undef)
          BT_objc_subscript_undef.reset(new BuiltinBug("Subscript access on an "
                                                       "uninitialized object "
                                                       "pointer"));
        BT = BT_objc_subscript_undef.get();
        break;
      }
      assert(BT && "Unknown message kind.");

      BugReport *R = new BugReport(*BT, BT->getName(), N);
      const ObjCMessageExpr *ME = msg.getOriginExpr();
      R->addRange(ME->getReceiverRange());

      // FIXME: getTrackNullOrUndefValueVisitor can't handle "super" yet.
      if (const Expr *ReceiverE = ME->getInstanceReceiver())
        bugreporter::trackNullOrUndefValue(N, ReceiverE, *R);
      C.emitReport(R);
    }
    return;
  } else {
    // Bifurcate the state into nil and non-nil ones.
    DefinedOrUnknownSVal receiverVal = cast<DefinedOrUnknownSVal>(recVal);

    ProgramStateRef state = C.getState();
    ProgramStateRef notNilState, nilState;
    llvm::tie(notNilState, nilState) = state->assume(receiverVal);

    // Handle receiver must be nil.
    if (nilState && !notNilState) {
      HandleNilReceiver(C, state, msg);
      return;
    }
  }
}

void CallAndMessageChecker::emitNilReceiverBug(CheckerContext &C,
                                               const ObjCMethodCall &msg,
                                               ExplodedNode *N) const {

  if (!BT_msg_ret)
    BT_msg_ret.reset(
      new BuiltinBug("Receiver in message expression is "
                     "'nil' and returns a garbage value"));

  const ObjCMessageExpr *ME = msg.getOriginExpr();

  SmallString<200> buf;
  llvm::raw_svector_ostream os(buf);
  os << "The receiver of message '" << ME->getSelector().getAsString()
     << "' is nil and returns a value of type '";
  msg.getResultType().print(os, C.getLangOpts());
  os << "' that will be garbage";

  BugReport *report = new BugReport(*BT_msg_ret, os.str(), N);
  report->addRange(ME->getReceiverRange());
  // FIXME: This won't track "self" in messages to super.
  if (const Expr *receiver = ME->getInstanceReceiver()) {
    bugreporter::trackNullOrUndefValue(N, receiver, *report);
  }
  C.emitReport(report);
}

static bool supportsNilWithFloatRet(const llvm::Triple &triple) {
  return (triple.getVendor() == llvm::Triple::Apple &&
          (triple.getOS() == llvm::Triple::IOS ||
           !triple.isMacOSXVersionLT(10,5)));
}

void CallAndMessageChecker::HandleNilReceiver(CheckerContext &C,
                                              ProgramStateRef state,
                                              const ObjCMethodCall &Msg) const {
  ASTContext &Ctx = C.getASTContext();

  // Check the return type of the message expression.  A message to nil will
  // return different values depending on the return type and the architecture.
  QualType RetTy = Msg.getResultType();
  CanQualType CanRetTy = Ctx.getCanonicalType(RetTy);
  const LocationContext *LCtx = C.getLocationContext();

  if (CanRetTy->isStructureOrClassType()) {
    // Structure returns are safe since the compiler zeroes them out.
    SVal V = C.getSValBuilder().makeZeroVal(RetTy);
    C.addTransition(state->BindExpr(Msg.getOriginExpr(), LCtx, V));
    return;
  }

  // Other cases: check if sizeof(return type) > sizeof(void*)
  if (CanRetTy != Ctx.VoidTy && C.getLocationContext()->getParentMap()
                                  .isConsumedExpr(Msg.getOriginExpr())) {
    // Compute: sizeof(void *) and sizeof(return type)
    const uint64_t voidPtrSize = Ctx.getTypeSize(Ctx.VoidPtrTy);
    const uint64_t returnTypeSize = Ctx.getTypeSize(CanRetTy);

    if (voidPtrSize < returnTypeSize &&
        !(supportsNilWithFloatRet(Ctx.getTargetInfo().getTriple()) &&
          (Ctx.FloatTy == CanRetTy ||
           Ctx.DoubleTy == CanRetTy ||
           Ctx.LongDoubleTy == CanRetTy ||
           Ctx.LongLongTy == CanRetTy ||
           Ctx.UnsignedLongLongTy == CanRetTy))) {
      if (ExplodedNode *N = C.generateSink(state))
        emitNilReceiverBug(C, Msg, N);
      return;
    }

    // Handle the safe cases where the return value is 0 if the
    // receiver is nil.
    //
    // FIXME: For now take the conservative approach that we only
    // return null values if we *know* that the receiver is nil.
    // This is because we can have surprises like:
    //
    //   ... = [[NSScreens screens] objectAtIndex:0];
    //
    // What can happen is that [... screens] could return nil, but
    // it most likely isn't nil.  We should assume the semantics
    // of this case unless we have *a lot* more knowledge.
    //
    SVal V = C.getSValBuilder().makeZeroVal(RetTy);
    C.addTransition(state->BindExpr(Msg.getOriginExpr(), LCtx, V));
    return;
  }

  C.addTransition(state);
}

void ento::registerCallAndMessageChecker(CheckerManager &mgr) {
  mgr.registerChecker<CallAndMessageChecker>();
}
