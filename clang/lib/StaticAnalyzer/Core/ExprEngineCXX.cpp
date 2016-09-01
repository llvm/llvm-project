//===- ExprEngineCXX.cpp - ExprEngine support for C++ -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the C++ expression evaluation engine.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"

using namespace clang;
using namespace ento;

void ExprEngine::CreateCXXTemporaryObject(const MaterializeTemporaryExpr *ME,
                                          ExplodedNode *Pred,
                                          ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);
  const Expr *tempExpr = ME->GetTemporaryExpr()->IgnoreParens();
  ProgramStateRef state = Pred->getState();
  const LocationContext *LCtx = Pred->getLocationContext();

  state = createTemporaryRegionIfNeeded(state, LCtx, tempExpr, ME);
  Bldr.generateNode(ME, Pred, state);
}

// FIXME: This is the sort of code that should eventually live in a Core
// checker rather than as a special case in ExprEngine.
void ExprEngine::performTrivialCopy(NodeBuilder &Bldr, ExplodedNode *Pred,
                                    const CallEvent &Call) {
  SVal ThisVal;
  bool AlwaysReturnsLValue;
  if (const CXXConstructorCall *Ctor = dyn_cast<CXXConstructorCall>(&Call)) {
    assert(Ctor->getDecl()->isTrivial());
    assert(Ctor->getDecl()->isCopyOrMoveConstructor());
    ThisVal = Ctor->getCXXThisVal();
    AlwaysReturnsLValue = false;
  } else {
    assert(cast<CXXMethodDecl>(Call.getDecl())->isTrivial());
    assert(cast<CXXMethodDecl>(Call.getDecl())->getOverloadedOperator() ==
           OO_Equal);
    ThisVal = cast<CXXInstanceCall>(Call).getCXXThisVal();
    AlwaysReturnsLValue = true;
  }

  const LocationContext *LCtx = Pred->getLocationContext();

  ExplodedNodeSet Dst;
  Bldr.takeNodes(Pred);

  SVal V = Call.getArgSVal(0);

  // If the value being copied is not unknown, load from its location to get
  // an aggregate rvalue.
  if (Optional<Loc> L = V.getAs<Loc>())
    V = Pred->getState()->getSVal(*L);
  else
    assert(V.isUnknown());

  const Expr *CallExpr = Call.getOriginExpr();
  evalBind(Dst, CallExpr, Pred, ThisVal, V, true);

  PostStmt PS(CallExpr, LCtx);
  for (ExplodedNodeSet::iterator I = Dst.begin(), E = Dst.end();
       I != E; ++I) {
    ProgramStateRef State = (*I)->getState();
    if (AlwaysReturnsLValue)
      State = State->BindExpr(CallExpr, LCtx, ThisVal);
    else
      State = bindReturnValue(Call, LCtx, State);
    Bldr.generateNode(PS, State, *I);
  }
}


/// Returns a region representing the first element of a (possibly
/// multi-dimensional) array.
///
/// On return, \p Ty will be set to the base type of the array.
///
/// If the type is not an array type at all, the original value is returned.
static SVal makeZeroElementRegion(ProgramStateRef State, SVal LValue,
                                  QualType &Ty) {
  SValBuilder &SVB = State->getStateManager().getSValBuilder();
  ASTContext &Ctx = SVB.getContext();

  while (const ArrayType *AT = Ctx.getAsArrayType(Ty)) {
    Ty = AT->getElementType();
    LValue = State->getLValue(Ty, SVB.makeZeroArrayIndex(), LValue);
  }

  return LValue;
}


const MemRegion *
ExprEngine::getRegionForConstructedObject(const CXXConstructExpr *CE,
                                          ExplodedNode *Pred) {
  const LocationContext *LCtx = Pred->getLocationContext();
  ProgramStateRef State = Pred->getState();

  // See if we're constructing an existing region by looking at the next
  // element in the CFG.

  if (auto Elem = findElementDirectlyInitializedByCurrentConstructor()) {
    if (Optional<CFGStmt> StmtElem = Elem->getAs<CFGStmt>()) {
      auto *DS = cast<DeclStmt>(StmtElem->getStmt());
      if (const auto *Var = dyn_cast<VarDecl>(DS->getSingleDecl())) {
        if (Var->getInit() && Var->getInit()->IgnoreImplicit() == CE) {
          SVal LValue = State->getLValue(Var, LCtx);
          QualType Ty = Var->getType();
          LValue = makeZeroElementRegion(State, LValue, Ty);
          return LValue.getAsRegion();
        }
      }
    } else if (Optional<CFGInitializer> InitElem = Elem->getAs<CFGInitializer>()) {
      const CXXCtorInitializer *Init = InitElem->getInitializer();
      assert(Init->isAnyMemberInitializer());
      const CXXMethodDecl *CurCtor = cast<CXXMethodDecl>(LCtx->getDecl());
      Loc ThisPtr =
      getSValBuilder().getCXXThis(CurCtor, LCtx->getCurrentStackFrame());
      SVal ThisVal = State->getSVal(ThisPtr);

      const ValueDecl *Field;
      SVal FieldVal;
      if (Init->isIndirectMemberInitializer()) {
        Field = Init->getIndirectMember();
        FieldVal = State->getLValue(Init->getIndirectMember(), ThisVal);
      } else {
        Field = Init->getMember();
        FieldVal = State->getLValue(Init->getMember(), ThisVal);
      }

      QualType Ty = Field->getType();
      FieldVal = makeZeroElementRegion(State, FieldVal, Ty);
      return FieldVal.getAsRegion();
    }

    // FIXME: This will eventually need to handle new-expressions as well.
    // Don't forget to update the pre-constructor initialization code in
    // ExprEngine::VisitCXXConstructExpr.
  }
  // If we couldn't find an existing region to construct into, assume we're
  // constructing a temporary.
  MemRegionManager &MRMgr = getSValBuilder().getRegionManager();
  return MRMgr.getCXXTempObjectRegion(CE, LCtx);
}

/// Returns true if the initializer for \Elem can be a direct
/// constructor.
static bool canHaveDirectConstructor(CFGElement Elem){
  // DeclStmts and CXXCtorInitializers for fields can be directly constructed.

  if (Optional<CFGStmt> StmtElem = Elem.getAs<CFGStmt>()) {
    if (isa<DeclStmt>(StmtElem->getStmt())) {
      return true;
    }
  }

  if (Elem.getKind() == CFGElement::Initializer) {
    return true;
  }

  return false;
}

Optional<CFGElement>
ExprEngine::findElementDirectlyInitializedByCurrentConstructor() {
  const NodeBuilderContext &CurrBldrCtx = getBuilderContext();
  // See if we're constructing an existing region by looking at the next
  // element in the CFG.
  const CFGBlock *B = CurrBldrCtx.getBlock();
  assert(isa<CXXConstructExpr>(((*B)[currStmtIdx]).castAs<CFGStmt>().getStmt()));
  unsigned int NextStmtIdx = currStmtIdx + 1;
  if (NextStmtIdx >= B->size())
    return None;

  CFGElement Next = (*B)[NextStmtIdx];

  // Is this a destructor? If so, we might be in the middle of an assignment
  // to a local or member: look ahead one more element to see what we find.
  while (Next.getAs<CFGImplicitDtor>() && NextStmtIdx + 1 < B->size()) {
    ++NextStmtIdx;
    Next = (*B)[NextStmtIdx];
  }

  if (canHaveDirectConstructor(Next))
    return Next;

  return None;
}

const CXXConstructExpr *
ExprEngine::findDirectConstructorForCurrentCFGElement() {
  // Go backward in the CFG to see if the previous element (ignoring
  // destructors) was a CXXConstructExpr. If so, that constructor
  // was constructed directly into an existing region.
  // This process is essentially the inverse of that performed in
  // findElementDirectlyInitializedByCurrentConstructor().
  if (currStmtIdx == 0)
    return nullptr;

  const CFGBlock *B = getBuilderContext().getBlock();
  assert(canHaveDirectConstructor((*B)[currStmtIdx]));

  unsigned int PreviousStmtIdx = currStmtIdx - 1;
  CFGElement Previous = (*B)[PreviousStmtIdx];

  while (Previous.getAs<CFGImplicitDtor>() && PreviousStmtIdx > 0) {
    --PreviousStmtIdx;
    Previous = (*B)[PreviousStmtIdx];
  }

  if (Optional<CFGStmt> PrevStmtElem = Previous.getAs<CFGStmt>()) {
    if (auto *CtorExpr = dyn_cast<CXXConstructExpr>(PrevStmtElem->getStmt())) {
      return CtorExpr;
    }
  }

  return nullptr;
}

void ExprEngine::VisitCXXConstructExpr(const CXXConstructExpr *CE,
                                       ExplodedNode *Pred,
                                       ExplodedNodeSet &destNodes) {
  const LocationContext *LCtx = Pred->getLocationContext();
  ProgramStateRef State = Pred->getState();

  const MemRegion *Target = nullptr;

  // FIXME: Handle arrays, which run the same constructor for every element.
  // For now, we just run the first constructor (which should still invalidate
  // the entire array).

  switch (CE->getConstructionKind()) {
  case CXXConstructExpr::CK_Complete: {
    Target = getRegionForConstructedObject(CE, Pred);
    break;
  }
  case CXXConstructExpr::CK_VirtualBase:
    // Make sure we are not calling virtual base class initializers twice.
    // Only the most-derived object should initialize virtual base classes.
    if (const Stmt *Outer = LCtx->getCurrentStackFrame()->getCallSite()) {
      const CXXConstructExpr *OuterCtor = dyn_cast<CXXConstructExpr>(Outer);
      if (OuterCtor) {
        switch (OuterCtor->getConstructionKind()) {
        case CXXConstructExpr::CK_NonVirtualBase:
        case CXXConstructExpr::CK_VirtualBase:
          // Bail out!
          destNodes.Add(Pred);
          return;
        case CXXConstructExpr::CK_Complete:
        case CXXConstructExpr::CK_Delegating:
          break;
        }
      }
    }
    // FALLTHROUGH
  case CXXConstructExpr::CK_NonVirtualBase:
  case CXXConstructExpr::CK_Delegating: {
    const CXXMethodDecl *CurCtor = cast<CXXMethodDecl>(LCtx->getDecl());
    Loc ThisPtr = getSValBuilder().getCXXThis(CurCtor,
                                              LCtx->getCurrentStackFrame());
    SVal ThisVal = State->getSVal(ThisPtr);

    if (CE->getConstructionKind() == CXXConstructExpr::CK_Delegating) {
      Target = ThisVal.getAsRegion();
    } else {
      // Cast to the base type.
      bool IsVirtual =
        (CE->getConstructionKind() == CXXConstructExpr::CK_VirtualBase);
      SVal BaseVal = getStoreManager().evalDerivedToBase(ThisVal, CE->getType(),
                                                         IsVirtual);
      Target = BaseVal.getAsRegion();
    }
    break;
  }
  }

  CallEventManager &CEMgr = getStateManager().getCallEventManager();
  CallEventRef<CXXConstructorCall> Call =
    CEMgr.getCXXConstructorCall(CE, Target, State, LCtx);

  ExplodedNodeSet DstPreVisit;
  getCheckerManager().runCheckersForPreStmt(DstPreVisit, Pred, CE, *this);

  ExplodedNodeSet PreInitialized;
  {
    StmtNodeBuilder Bldr(DstPreVisit, PreInitialized, *currBldrCtx);
    if (CE->requiresZeroInitialization()) {
      // Type of the zero doesn't matter.
      SVal ZeroVal = svalBuilder.makeZeroVal(getContext().CharTy);

      for (ExplodedNodeSet::iterator I = DstPreVisit.begin(),
                                     E = DstPreVisit.end();
           I != E; ++I) {
        ProgramStateRef State = (*I)->getState();
        // FIXME: Once we properly handle constructors in new-expressions, we'll
        // need to invalidate the region before setting a default value, to make
        // sure there aren't any lingering bindings around. This probably needs
        // to happen regardless of whether or not the object is zero-initialized
        // to handle random fields of a placement-initialized object picking up
        // old bindings. We might only want to do it when we need to, though.
        // FIXME: This isn't actually correct for arrays -- we need to zero-
        // initialize the entire array, not just the first element -- but our
        // handling of arrays everywhere else is weak as well, so this shouldn't
        // actually make things worse. Placement new makes this tricky as well,
        // since it's then possible to be initializing one part of a multi-
        // dimensional array.
        State = State->bindDefault(loc::MemRegionVal(Target), ZeroVal);
        Bldr.generateNode(CE, *I, State, /*tag=*/nullptr,
                          ProgramPoint::PreStmtKind);
      }
    }
  }

  ExplodedNodeSet DstPreCall;
  getCheckerManager().runCheckersForPreCall(DstPreCall, PreInitialized,
                                            *Call, *this);

  ExplodedNodeSet DstEvaluated;
  StmtNodeBuilder Bldr(DstPreCall, DstEvaluated, *currBldrCtx);

  bool IsArray = isa<ElementRegion>(Target);
  if (CE->getConstructor()->isTrivial() &&
      CE->getConstructor()->isCopyOrMoveConstructor() &&
      !IsArray) {
    // FIXME: Handle other kinds of trivial constructors as well.
    for (ExplodedNodeSet::iterator I = DstPreCall.begin(), E = DstPreCall.end();
         I != E; ++I)
      performTrivialCopy(Bldr, *I, *Call);

  } else {
    for (ExplodedNodeSet::iterator I = DstPreCall.begin(), E = DstPreCall.end();
         I != E; ++I)
      defaultEvalCall(Bldr, *I, *Call);
  }

  ExplodedNodeSet DstPostCall;
  getCheckerManager().runCheckersForPostCall(DstPostCall, DstEvaluated,
                                             *Call, *this);
  getCheckerManager().runCheckersForPostStmt(destNodes, DstPostCall, CE, *this);
}

void ExprEngine::VisitCXXDestructor(QualType ObjectType,
                                    const MemRegion *Dest,
                                    const Stmt *S,
                                    bool IsBaseDtor,
                                    ExplodedNode *Pred,
                                    ExplodedNodeSet &Dst) {
  const LocationContext *LCtx = Pred->getLocationContext();
  ProgramStateRef State = Pred->getState();

  // FIXME: We need to run the same destructor on every element of the array.
  // This workaround will just run the first destructor (which will still
  // invalidate the entire array).
  SVal DestVal = UnknownVal();
  if (Dest)
    DestVal = loc::MemRegionVal(Dest);
  DestVal = makeZeroElementRegion(State, DestVal, ObjectType);
  Dest = DestVal.getAsRegion();

  const CXXRecordDecl *RecordDecl = ObjectType->getAsCXXRecordDecl();
  assert(RecordDecl && "Only CXXRecordDecls should have destructors");
  const CXXDestructorDecl *DtorDecl = RecordDecl->getDestructor();

  CallEventManager &CEMgr = getStateManager().getCallEventManager();
  CallEventRef<CXXDestructorCall> Call =
    CEMgr.getCXXDestructorCall(DtorDecl, S, Dest, IsBaseDtor, State, LCtx);

  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                Call->getSourceRange().getBegin(),
                                "Error evaluating destructor");

  ExplodedNodeSet DstPreCall;
  getCheckerManager().runCheckersForPreCall(DstPreCall, Pred,
                                            *Call, *this);

  ExplodedNodeSet DstInvalidated;
  StmtNodeBuilder Bldr(DstPreCall, DstInvalidated, *currBldrCtx);
  for (ExplodedNodeSet::iterator I = DstPreCall.begin(), E = DstPreCall.end();
       I != E; ++I)
    defaultEvalCall(Bldr, *I, *Call);

  ExplodedNodeSet DstPostCall;
  getCheckerManager().runCheckersForPostCall(Dst, DstInvalidated,
                                             *Call, *this);
}

void ExprEngine::VisitCXXNewAllocatorCall(const CXXNewExpr *CNE,
                                          ExplodedNode *Pred,
                                          ExplodedNodeSet &Dst) {
  ProgramStateRef State = Pred->getState();
  const LocationContext *LCtx = Pred->getLocationContext();
  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                CNE->getStartLoc(),
                                "Error evaluating New Allocator Call");
  CallEventManager &CEMgr = getStateManager().getCallEventManager();
  CallEventRef<CXXAllocatorCall> Call =
    CEMgr.getCXXAllocatorCall(CNE, State, LCtx);

  ExplodedNodeSet DstPreCall;
  getCheckerManager().runCheckersForPreCall(DstPreCall, Pred,
                                            *Call, *this);

  ExplodedNodeSet DstInvalidated;
  StmtNodeBuilder Bldr(DstPreCall, DstInvalidated, *currBldrCtx);
  for (ExplodedNodeSet::iterator I = DstPreCall.begin(), E = DstPreCall.end();
       I != E; ++I)
    defaultEvalCall(Bldr, *I, *Call);
  getCheckerManager().runCheckersForPostCall(Dst, DstInvalidated,
                                             *Call, *this);
}


void ExprEngine::VisitCXXNewExpr(const CXXNewExpr *CNE, ExplodedNode *Pred,
                                   ExplodedNodeSet &Dst) {
  // FIXME: Much of this should eventually migrate to CXXAllocatorCall.
  // Also, we need to decide how allocators actually work -- they're not
  // really part of the CXXNewExpr because they happen BEFORE the
  // CXXConstructExpr subexpression. See PR12014 for some discussion.

  unsigned blockCount = currBldrCtx->blockCount();
  const LocationContext *LCtx = Pred->getLocationContext();
  DefinedOrUnknownSVal symVal = UnknownVal();
  FunctionDecl *FD = CNE->getOperatorNew();

  bool IsStandardGlobalOpNewFunction = false;
  if (FD && !isa<CXXMethodDecl>(FD) && !FD->isVariadic()) {
    if (FD->getNumParams() == 2) {
      QualType T = FD->getParamDecl(1)->getType();
      if (const IdentifierInfo *II = T.getBaseTypeIdentifier())
        // NoThrow placement new behaves as a standard new.
        IsStandardGlobalOpNewFunction = II->getName().equals("nothrow_t");
    }
    else
      // Placement forms are considered non-standard.
      IsStandardGlobalOpNewFunction = (FD->getNumParams() == 1);
  }

  // We assume all standard global 'operator new' functions allocate memory in
  // heap. We realize this is an approximation that might not correctly model
  // a custom global allocator.
  if (IsStandardGlobalOpNewFunction)
    symVal = svalBuilder.getConjuredHeapSymbolVal(CNE, LCtx, blockCount);
  else
    symVal = svalBuilder.conjureSymbolVal(nullptr, CNE, LCtx, CNE->getType(),
                                          blockCount);

  ProgramStateRef State = Pred->getState();
  CallEventManager &CEMgr = getStateManager().getCallEventManager();
  CallEventRef<CXXAllocatorCall> Call =
    CEMgr.getCXXAllocatorCall(CNE, State, LCtx);

  // Invalidate placement args.
  // FIXME: Once we figure out how we want allocators to work,
  // we should be using the usual pre-/(default-)eval-/post-call checks here.
  State = Call->invalidateRegions(blockCount);
  if (!State)
    return;

  // If this allocation function is not declared as non-throwing, failures
  // /must/ be signalled by exceptions, and thus the return value will never be
  // NULL. -fno-exceptions does not influence this semantics.
  // FIXME: GCC has a -fcheck-new option, which forces it to consider the case
  // where new can return NULL. If we end up supporting that option, we can
  // consider adding a check for it here.
  // C++11 [basic.stc.dynamic.allocation]p3.
  if (FD) {
    QualType Ty = FD->getType();
    if (const FunctionProtoType *ProtoType = Ty->getAs<FunctionProtoType>())
      if (!ProtoType->isNothrow(getContext()))
        State = State->assume(symVal, true);
  }

  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);

  if (CNE->isArray()) {
    // FIXME: allocating an array requires simulating the constructors.
    // For now, just return a symbolicated region.
    const MemRegion *NewReg = symVal.castAs<loc::MemRegionVal>().getRegion();
    QualType ObjTy = CNE->getType()->getAs<PointerType>()->getPointeeType();
    const ElementRegion *EleReg =
      getStoreManager().GetElementZeroRegion(NewReg, ObjTy);
    State = State->BindExpr(CNE, Pred->getLocationContext(),
                            loc::MemRegionVal(EleReg));
    Bldr.generateNode(CNE, Pred, State);
    return;
  }

  // FIXME: Once we have proper support for CXXConstructExprs inside
  // CXXNewExpr, we need to make sure that the constructed object is not
  // immediately invalidated here. (The placement call should happen before
  // the constructor call anyway.)
  SVal Result = symVal;
  if (FD && FD->isReservedGlobalPlacementOperator()) {
    // Non-array placement new should always return the placement location.
    SVal PlacementLoc = State->getSVal(CNE->getPlacementArg(0), LCtx);
    Result = svalBuilder.evalCast(PlacementLoc, CNE->getType(),
                                  CNE->getPlacementArg(0)->getType());
  }

  // Bind the address of the object, then check to see if we cached out.
  State = State->BindExpr(CNE, LCtx, Result);
  ExplodedNode *NewN = Bldr.generateNode(CNE, Pred, State);
  if (!NewN)
    return;

  // If the type is not a record, we won't have a CXXConstructExpr as an
  // initializer. Copy the value over.
  if (const Expr *Init = CNE->getInitializer()) {
    if (!isa<CXXConstructExpr>(Init)) {
      assert(Bldr.getResults().size() == 1);
      Bldr.takeNodes(NewN);
      evalBind(Dst, CNE, NewN, Result, State->getSVal(Init, LCtx),
               /*FirstInit=*/IsStandardGlobalOpNewFunction);
    }
  }
}

void ExprEngine::VisitCXXDeleteExpr(const CXXDeleteExpr *CDE,
                                    ExplodedNode *Pred, ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);
  ProgramStateRef state = Pred->getState();
  Bldr.generateNode(CDE, Pred, state);
}

void ExprEngine::VisitCXXCatchStmt(const CXXCatchStmt *CS,
                                   ExplodedNode *Pred,
                                   ExplodedNodeSet &Dst) {
  const VarDecl *VD = CS->getExceptionDecl();
  if (!VD) {
    Dst.Add(Pred);
    return;
  }

  const LocationContext *LCtx = Pred->getLocationContext();
  SVal V = svalBuilder.conjureSymbolVal(CS, LCtx, VD->getType(),
                                        currBldrCtx->blockCount());
  ProgramStateRef state = Pred->getState();
  state = state->bindLoc(state->getLValue(VD, LCtx), V);

  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);
  Bldr.generateNode(CS, Pred, state);
}

void ExprEngine::VisitCXXThisExpr(const CXXThisExpr *TE, ExplodedNode *Pred,
                                    ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);

  // Get the this object region from StoreManager.
  const LocationContext *LCtx = Pred->getLocationContext();
  const MemRegion *R =
    svalBuilder.getRegionManager().getCXXThisRegion(
                                  getContext().getCanonicalType(TE->getType()),
                                                    LCtx);

  ProgramStateRef state = Pred->getState();
  SVal V = state->getSVal(loc::MemRegionVal(R));
  Bldr.generateNode(TE, Pred, state->BindExpr(TE, LCtx, V));
}

void ExprEngine::VisitLambdaExpr(const LambdaExpr *LE, ExplodedNode *Pred,
                                 ExplodedNodeSet &Dst) {
  const LocationContext *LocCtxt = Pred->getLocationContext();

  // Get the region of the lambda itself.
  const MemRegion *R = svalBuilder.getRegionManager().getCXXTempObjectRegion(
      LE, LocCtxt);
  SVal V = loc::MemRegionVal(R);

  ProgramStateRef State = Pred->getState();

  // If we created a new MemRegion for the lambda, we should explicitly bind
  // the captures.
  CXXRecordDecl::field_iterator CurField = LE->getLambdaClass()->field_begin();
  for (LambdaExpr::const_capture_init_iterator i = LE->capture_init_begin(),
                                               e = LE->capture_init_end();
       i != e; ++i, ++CurField) {
    FieldDecl *FieldForCapture = *CurField;
    SVal FieldLoc = State->getLValue(FieldForCapture, V);

    SVal InitVal;
    if (!FieldForCapture->hasCapturedVLAType()) {
      Expr *InitExpr = *i;
      assert(InitExpr && "Capture missing initialization expression");
      InitVal = State->getSVal(InitExpr, LocCtxt);
    } else {
      // The field stores the length of a captured variable-length array.
      // These captures don't have initialization expressions; instead we
      // get the length from the VLAType size expression.
      Expr *SizeExpr = FieldForCapture->getCapturedVLAType()->getSizeExpr();
      InitVal = State->getSVal(SizeExpr, LocCtxt);
    }

    State = State->bindLoc(FieldLoc, InitVal);
  }

  // Decay the Loc into an RValue, because there might be a
  // MaterializeTemporaryExpr node above this one which expects the bound value
  // to be an RValue.
  SVal LambdaRVal = State->getSVal(R);

  ExplodedNodeSet Tmp;
  StmtNodeBuilder Bldr(Pred, Tmp, *currBldrCtx);
  // FIXME: is this the right program point kind?
  Bldr.generateNode(LE, Pred,
                    State->BindExpr(LE, LocCtxt, LambdaRVal),
                    nullptr, ProgramPoint::PostLValueKind);

  // FIXME: Move all post/pre visits to ::Visit().
  getCheckerManager().runCheckersForPostStmt(Dst, Tmp, LE, *this);
}
