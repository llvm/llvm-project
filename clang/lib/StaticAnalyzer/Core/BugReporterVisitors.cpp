// BugReporterVisitors.cpp - Helpers for reporting bugs -----------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a set of BugReporter "visitors" which can be used to
//  enhance the diagnostics reported for a bug.
//
//===----------------------------------------------------------------------===//
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

using llvm::FoldingSetNodeID;

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

bool bugreporter::isDeclRefExprToReference(const Expr *E) {
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    return DRE->getDecl()->getType()->isReferenceType();
  }
  return false;
}

const Expr *bugreporter::getDerefExpr(const Stmt *S) {
  // Pattern match for a few useful cases:
  //   a[0], p->f, *p
  const Expr *E = dyn_cast<Expr>(S);
  if (!E)
    return nullptr;
  E = E->IgnoreParenCasts();

  while (true) {
    if (const BinaryOperator *B = dyn_cast<BinaryOperator>(E)) {
      assert(B->isAssignmentOp());
      E = B->getLHS()->IgnoreParenCasts();
      continue;
    }
    else if (const UnaryOperator *U = dyn_cast<UnaryOperator>(E)) {
      if (U->getOpcode() == UO_Deref)
        return U->getSubExpr()->IgnoreParenCasts();
    }
    else if (const MemberExpr *ME = dyn_cast<MemberExpr>(E)) {
      if (ME->isArrow() || isDeclRefExprToReference(ME->getBase())) {
        return ME->getBase()->IgnoreParenCasts();
      } else {
        // If we have a member expr with a dot, the base must have been
        // dereferenced.
        return getDerefExpr(ME->getBase());
      }
    }
    else if (const ObjCIvarRefExpr *IvarRef = dyn_cast<ObjCIvarRefExpr>(E)) {
      return IvarRef->getBase()->IgnoreParenCasts();
    }
    else if (const ArraySubscriptExpr *AE = dyn_cast<ArraySubscriptExpr>(E)) {
      return AE->getBase();
    }
    else if (isDeclRefExprToReference(E)) {
      return E;
    }
    break;
  }

  return nullptr;
}

const Stmt *bugreporter::GetDenomExpr(const ExplodedNode *N) {
  const Stmt *S = N->getLocationAs<PreStmt>()->getStmt();
  if (const BinaryOperator *BE = dyn_cast<BinaryOperator>(S))
    return BE->getRHS();
  return nullptr;
}

const Stmt *bugreporter::GetRetValExpr(const ExplodedNode *N) {
  const Stmt *S = N->getLocationAs<PostStmt>()->getStmt();
  if (const ReturnStmt *RS = dyn_cast<ReturnStmt>(S))
    return RS->getRetValue();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Definitions for bug reporter visitors.
//===----------------------------------------------------------------------===//

std::unique_ptr<PathDiagnosticPiece>
BugReporterVisitor::getEndPath(BugReporterContext &BRC,
                               const ExplodedNode *EndPathNode, BugReport &BR) {
  return nullptr;
}

std::unique_ptr<PathDiagnosticPiece> BugReporterVisitor::getDefaultEndPath(
    BugReporterContext &BRC, const ExplodedNode *EndPathNode, BugReport &BR) {
  PathDiagnosticLocation L =
    PathDiagnosticLocation::createEndOfPath(EndPathNode,BRC.getSourceManager());

  const auto &Ranges = BR.getRanges();

  // Only add the statement itself as a range if we didn't specify any
  // special ranges for this report.
  auto P = llvm::make_unique<PathDiagnosticEventPiece>(
      L, BR.getDescription(), Ranges.begin() == Ranges.end());
  for (SourceRange Range : Ranges)
    P->addRange(Range);

  return std::move(P);
}


namespace {
/// Emits an extra note at the return statement of an interesting stack frame.
///
/// The returned value is marked as an interesting value, and if it's null,
/// adds a visitor to track where it became null.
///
/// This visitor is intended to be used when another visitor discovers that an
/// interesting value comes from an inlined function call.
class ReturnVisitor : public BugReporterVisitorImpl<ReturnVisitor> {
  const StackFrameContext *StackFrame;
  enum {
    Initial,
    MaybeUnsuppress,
    Satisfied
  } Mode;

  bool EnableNullFPSuppression;

public:
  ReturnVisitor(const StackFrameContext *Frame, bool Suppressed)
    : StackFrame(Frame), Mode(Initial), EnableNullFPSuppression(Suppressed) {}

  static void *getTag() {
    static int Tag = 0;
    return static_cast<void *>(&Tag);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    ID.AddPointer(ReturnVisitor::getTag());
    ID.AddPointer(StackFrame);
    ID.AddBoolean(EnableNullFPSuppression);
  }

  /// Adds a ReturnVisitor if the given statement represents a call that was
  /// inlined.
  ///
  /// This will search back through the ExplodedGraph, starting from the given
  /// node, looking for when the given statement was processed. If it turns out
  /// the statement is a call that was inlined, we add the visitor to the
  /// bug report, so it can print a note later.
  static void addVisitorIfNecessary(const ExplodedNode *Node, const Stmt *S,
                                    BugReport &BR,
                                    bool InEnableNullFPSuppression) {
    if (!CallEvent::isCallStmt(S))
      return;

    // First, find when we processed the statement.
    do {
      if (Optional<CallExitEnd> CEE = Node->getLocationAs<CallExitEnd>())
        if (CEE->getCalleeContext()->getCallSite() == S)
          break;
      if (Optional<StmtPoint> SP = Node->getLocationAs<StmtPoint>())
        if (SP->getStmt() == S)
          break;

      Node = Node->getFirstPred();
    } while (Node);

    // Next, step over any post-statement checks.
    while (Node && Node->getLocation().getAs<PostStmt>())
      Node = Node->getFirstPred();
    if (!Node)
      return;

    // Finally, see if we inlined the call.
    Optional<CallExitEnd> CEE = Node->getLocationAs<CallExitEnd>();
    if (!CEE)
      return;

    const StackFrameContext *CalleeContext = CEE->getCalleeContext();
    if (CalleeContext->getCallSite() != S)
      return;

    // Check the return value.
    ProgramStateRef State = Node->getState();
    SVal RetVal = State->getSVal(S, Node->getLocationContext());

    // Handle cases where a reference is returned and then immediately used.
    if (cast<Expr>(S)->isGLValue())
      if (Optional<Loc> LValue = RetVal.getAs<Loc>())
        RetVal = State->getSVal(*LValue);

    // See if the return value is NULL. If so, suppress the report.
    SubEngine *Eng = State->getStateManager().getOwningEngine();
    assert(Eng && "Cannot file a bug report without an owning engine");
    AnalyzerOptions &Options = Eng->getAnalysisManager().options;

    bool EnableNullFPSuppression = false;
    if (InEnableNullFPSuppression && Options.shouldSuppressNullReturnPaths())
      if (Optional<Loc> RetLoc = RetVal.getAs<Loc>())
        EnableNullFPSuppression = State->isNull(*RetLoc).isConstrainedTrue();

    BR.markInteresting(CalleeContext);
    BR.addVisitor(llvm::make_unique<ReturnVisitor>(CalleeContext,
                                                   EnableNullFPSuppression));
  }

  /// Returns true if any counter-suppression heuristics are enabled for
  /// ReturnVisitor.
  static bool hasCounterSuppression(AnalyzerOptions &Options) {
    return Options.shouldAvoidSuppressingNullArgumentPaths();
  }

  PathDiagnosticPiece *visitNodeInitial(const ExplodedNode *N,
                                        const ExplodedNode *PrevN,
                                        BugReporterContext &BRC,
                                        BugReport &BR) {
    // Only print a message at the interesting return statement.
    if (N->getLocationContext() != StackFrame)
      return nullptr;

    Optional<StmtPoint> SP = N->getLocationAs<StmtPoint>();
    if (!SP)
      return nullptr;

    const ReturnStmt *Ret = dyn_cast<ReturnStmt>(SP->getStmt());
    if (!Ret)
      return nullptr;

    // Okay, we're at the right return statement, but do we have the return
    // value available?
    ProgramStateRef State = N->getState();
    SVal V = State->getSVal(Ret, StackFrame);
    if (V.isUnknownOrUndef())
      return nullptr;

    // Don't print any more notes after this one.
    Mode = Satisfied;

    const Expr *RetE = Ret->getRetValue();
    assert(RetE && "Tracking a return value for a void function");

    // Handle cases where a reference is returned and then immediately used.
    Optional<Loc> LValue;
    if (RetE->isGLValue()) {
      if ((LValue = V.getAs<Loc>())) {
        SVal RValue = State->getRawSVal(*LValue, RetE->getType());
        if (RValue.getAs<DefinedSVal>())
          V = RValue;
      }
    }

    // Ignore aggregate rvalues.
    if (V.getAs<nonloc::LazyCompoundVal>() ||
        V.getAs<nonloc::CompoundVal>())
      return nullptr;

    RetE = RetE->IgnoreParenCasts();

    // If we can't prove the return value is 0, just mark it interesting, and
    // make sure to track it into any further inner functions.
    if (!State->isNull(V).isConstrainedTrue()) {
      BR.markInteresting(V);
      ReturnVisitor::addVisitorIfNecessary(N, RetE, BR,
                                           EnableNullFPSuppression);
      return nullptr;
    }

    // If we're returning 0, we should track where that 0 came from.
    bugreporter::trackNullOrUndefValue(N, RetE, BR, /*IsArg*/ false,
                                       EnableNullFPSuppression);

    // Build an appropriate message based on the return value.
    SmallString<64> Msg;
    llvm::raw_svector_ostream Out(Msg);

    if (V.getAs<Loc>()) {
      // If we have counter-suppression enabled, make sure we keep visiting
      // future nodes. We want to emit a path note as well, in case
      // the report is resurrected as valid later on.
      ExprEngine &Eng = BRC.getBugReporter().getEngine();
      AnalyzerOptions &Options = Eng.getAnalysisManager().options;
      if (EnableNullFPSuppression && hasCounterSuppression(Options))
        Mode = MaybeUnsuppress;

      if (RetE->getType()->isObjCObjectPointerType())
        Out << "Returning nil";
      else
        Out << "Returning null pointer";
    } else {
      Out << "Returning zero";
    }

    if (LValue) {
      if (const MemRegion *MR = LValue->getAsRegion()) {
        if (MR->canPrintPretty()) {
          Out << " (reference to ";
          MR->printPretty(Out);
          Out << ")";
        }
      }
    } else {
      // FIXME: We should have a more generalized location printing mechanism.
      if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(RetE))
        if (const DeclaratorDecl *DD = dyn_cast<DeclaratorDecl>(DR->getDecl()))
          Out << " (loaded from '" << *DD << "')";
    }

    PathDiagnosticLocation L(Ret, BRC.getSourceManager(), StackFrame);
    return new PathDiagnosticEventPiece(L, Out.str());
  }

  PathDiagnosticPiece *visitNodeMaybeUnsuppress(const ExplodedNode *N,
                                                const ExplodedNode *PrevN,
                                                BugReporterContext &BRC,
                                                BugReport &BR) {
#ifndef NDEBUG
    ExprEngine &Eng = BRC.getBugReporter().getEngine();
    AnalyzerOptions &Options = Eng.getAnalysisManager().options;
    assert(hasCounterSuppression(Options));
#endif

    // Are we at the entry node for this call?
    Optional<CallEnter> CE = N->getLocationAs<CallEnter>();
    if (!CE)
      return nullptr;

    if (CE->getCalleeContext() != StackFrame)
      return nullptr;

    Mode = Satisfied;

    // Don't automatically suppress a report if one of the arguments is
    // known to be a null pointer. Instead, start tracking /that/ null
    // value back to its origin.
    ProgramStateManager &StateMgr = BRC.getStateManager();
    CallEventManager &CallMgr = StateMgr.getCallEventManager();

    ProgramStateRef State = N->getState();
    CallEventRef<> Call = CallMgr.getCaller(StackFrame, State);
    for (unsigned I = 0, E = Call->getNumArgs(); I != E; ++I) {
      Optional<Loc> ArgV = Call->getArgSVal(I).getAs<Loc>();
      if (!ArgV)
        continue;

      const Expr *ArgE = Call->getArgExpr(I);
      if (!ArgE)
        continue;

      // Is it possible for this argument to be non-null?
      if (!State->isNull(*ArgV).isConstrainedTrue())
        continue;

      if (bugreporter::trackNullOrUndefValue(N, ArgE, BR, /*IsArg=*/true,
                                             EnableNullFPSuppression))
        BR.removeInvalidation(ReturnVisitor::getTag(), StackFrame);

      // If we /can't/ track the null pointer, we should err on the side of
      // false negatives, and continue towards marking this report invalid.
      // (We will still look at the other arguments, though.)
    }

    return nullptr;
  }

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC,
                                 BugReport &BR) override {
    switch (Mode) {
    case Initial:
      return visitNodeInitial(N, PrevN, BRC, BR);
    case MaybeUnsuppress:
      return visitNodeMaybeUnsuppress(N, PrevN, BRC, BR);
    case Satisfied:
      return nullptr;
    }

    llvm_unreachable("Invalid visit mode!");
  }

  std::unique_ptr<PathDiagnosticPiece> getEndPath(BugReporterContext &BRC,
                                                  const ExplodedNode *N,
                                                  BugReport &BR) override {
    if (EnableNullFPSuppression)
      BR.markInvalid(ReturnVisitor::getTag(), StackFrame);
    return nullptr;
  }
};
} // end anonymous namespace


void FindLastStoreBRVisitor ::Profile(llvm::FoldingSetNodeID &ID) const {
  static int tag = 0;
  ID.AddPointer(&tag);
  ID.AddPointer(R);
  ID.Add(V);
  ID.AddBoolean(EnableNullFPSuppression);
}

/// Returns true if \p N represents the DeclStmt declaring and initializing
/// \p VR.
static bool isInitializationOfVar(const ExplodedNode *N, const VarRegion *VR) {
  Optional<PostStmt> P = N->getLocationAs<PostStmt>();
  if (!P)
    return false;

  const DeclStmt *DS = P->getStmtAs<DeclStmt>();
  if (!DS)
    return false;

  if (DS->getSingleDecl() != VR->getDecl())
    return false;

  const MemSpaceRegion *VarSpace = VR->getMemorySpace();
  const StackSpaceRegion *FrameSpace = dyn_cast<StackSpaceRegion>(VarSpace);
  if (!FrameSpace) {
    // If we ever directly evaluate global DeclStmts, this assertion will be
    // invalid, but this still seems preferable to silently accepting an
    // initialization that may be for a path-sensitive variable.
    assert(VR->getDecl()->isStaticLocal() && "non-static stackless VarRegion");
    return true;
  }

  assert(VR->getDecl()->hasLocalStorage());
  const LocationContext *LCtx = N->getLocationContext();
  return FrameSpace->getStackFrame() == LCtx->getCurrentStackFrame();
}

PathDiagnosticPiece *FindLastStoreBRVisitor::VisitNode(const ExplodedNode *Succ,
                                                       const ExplodedNode *Pred,
                                                       BugReporterContext &BRC,
                                                       BugReport &BR) {

  if (Satisfied)
    return nullptr;

  const ExplodedNode *StoreSite = nullptr;
  const Expr *InitE = nullptr;
  bool IsParam = false;

  // First see if we reached the declaration of the region.
  if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
    if (isInitializationOfVar(Pred, VR)) {
      StoreSite = Pred;
      InitE = VR->getDecl()->getInit();
    }
  }

  // If this is a post initializer expression, initializing the region, we
  // should track the initializer expression.
  if (Optional<PostInitializer> PIP = Pred->getLocationAs<PostInitializer>()) {
    const MemRegion *FieldReg = (const MemRegion *)PIP->getLocationValue();
    if (FieldReg && FieldReg == R) {
      StoreSite = Pred;
      InitE = PIP->getInitializer()->getInit();
    }
  }

  // Otherwise, see if this is the store site:
  // (1) Succ has this binding and Pred does not, i.e. this is
  //     where the binding first occurred.
  // (2) Succ has this binding and is a PostStore node for this region, i.e.
  //     the same binding was re-assigned here.
  if (!StoreSite) {
    if (Succ->getState()->getSVal(R) != V)
      return nullptr;

    if (Pred->getState()->getSVal(R) == V) {
      Optional<PostStore> PS = Succ->getLocationAs<PostStore>();
      if (!PS || PS->getLocationValue() != R)
        return nullptr;
    }

    StoreSite = Succ;

    // If this is an assignment expression, we can track the value
    // being assigned.
    if (Optional<PostStmt> P = Succ->getLocationAs<PostStmt>())
      if (const BinaryOperator *BO = P->getStmtAs<BinaryOperator>())
        if (BO->isAssignmentOp())
          InitE = BO->getRHS();

    // If this is a call entry, the variable should be a parameter.
    // FIXME: Handle CXXThisRegion as well. (This is not a priority because
    // 'this' should never be NULL, but this visitor isn't just for NULL and
    // UndefinedVal.)
    if (Optional<CallEnter> CE = Succ->getLocationAs<CallEnter>()) {
      if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
        const ParmVarDecl *Param = cast<ParmVarDecl>(VR->getDecl());

        ProgramStateManager &StateMgr = BRC.getStateManager();
        CallEventManager &CallMgr = StateMgr.getCallEventManager();

        CallEventRef<> Call = CallMgr.getCaller(CE->getCalleeContext(),
                                                Succ->getState());
        InitE = Call->getArgExpr(Param->getFunctionScopeIndex());
        IsParam = true;
      }
    }

    // If this is a CXXTempObjectRegion, the Expr responsible for its creation
    // is wrapped inside of it.
    if (const CXXTempObjectRegion *TmpR = dyn_cast<CXXTempObjectRegion>(R))
      InitE = TmpR->getExpr();
  }

  if (!StoreSite)
    return nullptr;
  Satisfied = true;

  // If we have an expression that provided the value, try to track where it
  // came from.
  if (InitE) {
    if (V.isUndef() ||
        V.getAs<loc::ConcreteInt>() || V.getAs<nonloc::ConcreteInt>()) {
      if (!IsParam)
        InitE = InitE->IgnoreParenCasts();
      bugreporter::trackNullOrUndefValue(StoreSite, InitE, BR, IsParam,
                                         EnableNullFPSuppression);
    } else {
      ReturnVisitor::addVisitorIfNecessary(StoreSite, InitE->IgnoreParenCasts(),
                                           BR, EnableNullFPSuppression);
    }
  }

  // Okay, we've found the binding. Emit an appropriate message.
  SmallString<256> sbuf;
  llvm::raw_svector_ostream os(sbuf);

  if (Optional<PostStmt> PS = StoreSite->getLocationAs<PostStmt>()) {
    const Stmt *S = PS->getStmt();
    const char *action = nullptr;
    const DeclStmt *DS = dyn_cast<DeclStmt>(S);
    const VarRegion *VR = dyn_cast<VarRegion>(R);

    if (DS) {
      action = R->canPrintPretty() ? "initialized to " :
                                     "Initializing to ";
    } else if (isa<BlockExpr>(S)) {
      action = R->canPrintPretty() ? "captured by block as " :
                                     "Captured by block as ";
      if (VR) {
        // See if we can get the BlockVarRegion.
        ProgramStateRef State = StoreSite->getState();
        SVal V = State->getSVal(S, PS->getLocationContext());
        if (const BlockDataRegion *BDR =
              dyn_cast_or_null<BlockDataRegion>(V.getAsRegion())) {
          if (const VarRegion *OriginalR = BDR->getOriginalRegion(VR)) {
            if (Optional<KnownSVal> KV =
                State->getSVal(OriginalR).getAs<KnownSVal>())
              BR.addVisitor(llvm::make_unique<FindLastStoreBRVisitor>(
                  *KV, OriginalR, EnableNullFPSuppression));
          }
        }
      }
    }

    if (action) {
      if (R->canPrintPretty()) {
        R->printPretty(os);
        os << " ";
      }

      if (V.getAs<loc::ConcreteInt>()) {
        bool b = false;
        if (R->isBoundable()) {
          if (const TypedValueRegion *TR = dyn_cast<TypedValueRegion>(R)) {
            if (TR->getValueType()->isObjCObjectPointerType()) {
              os << action << "nil";
              b = true;
            }
          }
        }

        if (!b)
          os << action << "a null pointer value";
      } else if (Optional<nonloc::ConcreteInt> CVal =
                     V.getAs<nonloc::ConcreteInt>()) {
        os << action << CVal->getValue();
      }
      else if (DS) {
        if (V.isUndef()) {
          if (isa<VarRegion>(R)) {
            const VarDecl *VD = cast<VarDecl>(DS->getSingleDecl());
            if (VD->getInit()) {
              os << (R->canPrintPretty() ? "initialized" : "Initializing")
                 << " to a garbage value";
            } else {
              os << (R->canPrintPretty() ? "declared" : "Declaring")
                 << " without an initial value";
            }
          }
        }
        else {
          os << (R->canPrintPretty() ? "initialized" : "Initialized")
             << " here";
        }
      }
    }
  } else if (StoreSite->getLocation().getAs<CallEnter>()) {
    if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
      const ParmVarDecl *Param = cast<ParmVarDecl>(VR->getDecl());

      os << "Passing ";

      if (V.getAs<loc::ConcreteInt>()) {
        if (Param->getType()->isObjCObjectPointerType())
          os << "nil object reference";
        else
          os << "null pointer value";
      } else if (V.isUndef()) {
        os << "uninitialized value";
      } else if (Optional<nonloc::ConcreteInt> CI =
                     V.getAs<nonloc::ConcreteInt>()) {
        os << "the value " << CI->getValue();
      } else {
        os << "value";
      }

      // Printed parameter indexes are 1-based, not 0-based.
      unsigned Idx = Param->getFunctionScopeIndex() + 1;
      os << " via " << Idx << llvm::getOrdinalSuffix(Idx) << " parameter";
      if (R->canPrintPretty()) {
        os << " ";
        R->printPretty(os);
      }
    }
  }

  if (os.str().empty()) {
    if (V.getAs<loc::ConcreteInt>()) {
      bool b = false;
      if (R->isBoundable()) {
        if (const TypedValueRegion *TR = dyn_cast<TypedValueRegion>(R)) {
          if (TR->getValueType()->isObjCObjectPointerType()) {
            os << "nil object reference stored";
            b = true;
          }
        }
      }
      if (!b) {
        if (R->canPrintPretty())
          os << "Null pointer value stored";
        else
          os << "Storing null pointer value";
      }

    } else if (V.isUndef()) {
      if (R->canPrintPretty())
        os << "Uninitialized value stored";
      else
        os << "Storing uninitialized value";

    } else if (Optional<nonloc::ConcreteInt> CV =
                   V.getAs<nonloc::ConcreteInt>()) {
      if (R->canPrintPretty())
        os << "The value " << CV->getValue() << " is assigned";
      else
        os << "Assigning " << CV->getValue();

    } else {
      if (R->canPrintPretty())
        os << "Value assigned";
      else
        os << "Assigning value";
    }

    if (R->canPrintPretty()) {
      os << " to ";
      R->printPretty(os);
    }
  }

  // Construct a new PathDiagnosticPiece.
  ProgramPoint P = StoreSite->getLocation();
  PathDiagnosticLocation L;
  if (P.getAs<CallEnter>() && InitE)
    L = PathDiagnosticLocation(InitE, BRC.getSourceManager(),
                               P.getLocationContext());

  if (!L.isValid() || !L.asLocation().isValid())
    L = PathDiagnosticLocation::create(P, BRC.getSourceManager());

  if (!L.isValid() || !L.asLocation().isValid())
    return nullptr;

  return new PathDiagnosticEventPiece(L, os.str());
}

void TrackConstraintBRVisitor::Profile(llvm::FoldingSetNodeID &ID) const {
  static int tag = 0;
  ID.AddPointer(&tag);
  ID.AddBoolean(Assumption);
  ID.Add(Constraint);
}

/// Return the tag associated with this visitor.  This tag will be used
/// to make all PathDiagnosticPieces created by this visitor.
const char *TrackConstraintBRVisitor::getTag() {
  return "TrackConstraintBRVisitor";
}

bool TrackConstraintBRVisitor::isUnderconstrained(const ExplodedNode *N) const {
  if (IsZeroCheck)
    return N->getState()->isNull(Constraint).isUnderconstrained();
  return (bool)N->getState()->assume(Constraint, !Assumption);
}

PathDiagnosticPiece *
TrackConstraintBRVisitor::VisitNode(const ExplodedNode *N,
                                    const ExplodedNode *PrevN,
                                    BugReporterContext &BRC,
                                    BugReport &BR) {
  if (IsSatisfied)
    return nullptr;

  // Start tracking after we see the first state in which the value is
  // constrained.
  if (!IsTrackingTurnedOn)
    if (!isUnderconstrained(N))
      IsTrackingTurnedOn = true;
  if (!IsTrackingTurnedOn)
    return nullptr;

  // Check if in the previous state it was feasible for this constraint
  // to *not* be true.
  if (isUnderconstrained(PrevN)) {

    IsSatisfied = true;

    // As a sanity check, make sure that the negation of the constraint
    // was infeasible in the current state.  If it is feasible, we somehow
    // missed the transition point.
    assert(!isUnderconstrained(N));

    // We found the transition point for the constraint.  We now need to
    // pretty-print the constraint. (work-in-progress)
    SmallString<64> sbuf;
    llvm::raw_svector_ostream os(sbuf);

    if (Constraint.getAs<Loc>()) {
      os << "Assuming pointer value is ";
      os << (Assumption ? "non-null" : "null");
    }

    if (os.str().empty())
      return nullptr;

    // Construct a new PathDiagnosticPiece.
    ProgramPoint P = N->getLocation();
    PathDiagnosticLocation L =
      PathDiagnosticLocation::create(P, BRC.getSourceManager());
    if (!L.isValid())
      return nullptr;

    PathDiagnosticEventPiece *X = new PathDiagnosticEventPiece(L, os.str());
    X->setTag(getTag());
    return X;
  }

  return nullptr;
}

SuppressInlineDefensiveChecksVisitor::
SuppressInlineDefensiveChecksVisitor(DefinedSVal Value, const ExplodedNode *N)
  : V(Value), IsSatisfied(false), IsTrackingTurnedOn(false) {

    // Check if the visitor is disabled.
    SubEngine *Eng = N->getState()->getStateManager().getOwningEngine();
    assert(Eng && "Cannot file a bug report without an owning engine");
    AnalyzerOptions &Options = Eng->getAnalysisManager().options;
    if (!Options.shouldSuppressInlinedDefensiveChecks())
      IsSatisfied = true;

    assert(N->getState()->isNull(V).isConstrainedTrue() &&
           "The visitor only tracks the cases where V is constrained to 0");
}

void SuppressInlineDefensiveChecksVisitor::Profile(FoldingSetNodeID &ID) const {
  static int id = 0;
  ID.AddPointer(&id);
  ID.Add(V);
}

const char *SuppressInlineDefensiveChecksVisitor::getTag() {
  return "IDCVisitor";
}

PathDiagnosticPiece *
SuppressInlineDefensiveChecksVisitor::VisitNode(const ExplodedNode *Succ,
                                                const ExplodedNode *Pred,
                                                BugReporterContext &BRC,
                                                BugReport &BR) {
  if (IsSatisfied)
    return nullptr;

  // Start tracking after we see the first state in which the value is null.
  if (!IsTrackingTurnedOn)
    if (Succ->getState()->isNull(V).isConstrainedTrue())
      IsTrackingTurnedOn = true;
  if (!IsTrackingTurnedOn)
    return nullptr;

  // Check if in the previous state it was feasible for this value
  // to *not* be null.
  if (!Pred->getState()->isNull(V).isConstrainedTrue()) {
    IsSatisfied = true;

    assert(Succ->getState()->isNull(V).isConstrainedTrue());

    // Check if this is inlined defensive checks.
    const LocationContext *CurLC =Succ->getLocationContext();
    const LocationContext *ReportLC = BR.getErrorNode()->getLocationContext();
    if (CurLC != ReportLC && !CurLC->isParentOf(ReportLC))
      BR.markInvalid("Suppress IDC", CurLC);
  }
  return nullptr;
}

static const MemRegion *getLocationRegionIfReference(const Expr *E,
                                                     const ExplodedNode *N) {
  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(E)) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
      if (!VD->getType()->isReferenceType())
        return nullptr;
      ProgramStateManager &StateMgr = N->getState()->getStateManager();
      MemRegionManager &MRMgr = StateMgr.getRegionManager();
      return MRMgr.getVarRegion(VD, N->getLocationContext());
    }
  }

  // FIXME: This does not handle other kinds of null references,
  // for example, references from FieldRegions:
  //   struct Wrapper { int &ref; };
  //   Wrapper w = { *(int *)0 };
  //   w.ref = 1;

  return nullptr;
}

static const Expr *peelOffOuterExpr(const Expr *Ex,
                                    const ExplodedNode *N) {
  Ex = Ex->IgnoreParenCasts();
  if (const ExprWithCleanups *EWC = dyn_cast<ExprWithCleanups>(Ex))
    return peelOffOuterExpr(EWC->getSubExpr(), N);
  if (const OpaqueValueExpr *OVE = dyn_cast<OpaqueValueExpr>(Ex))
    return peelOffOuterExpr(OVE->getSourceExpr(), N);

  // Peel off the ternary operator.
  if (const ConditionalOperator *CO = dyn_cast<ConditionalOperator>(Ex)) {
    // Find a node where the branching occurred and find out which branch
    // we took (true/false) by looking at the ExplodedGraph.
    const ExplodedNode *NI = N;
    do {
      ProgramPoint ProgPoint = NI->getLocation();
      if (Optional<BlockEdge> BE = ProgPoint.getAs<BlockEdge>()) {
        const CFGBlock *srcBlk = BE->getSrc();
        if (const Stmt *term = srcBlk->getTerminator()) {
          if (term == CO) {
            bool TookTrueBranch = (*(srcBlk->succ_begin()) == BE->getDst());
            if (TookTrueBranch)
              return peelOffOuterExpr(CO->getTrueExpr(), N);
            else
              return peelOffOuterExpr(CO->getFalseExpr(), N);
          }
        }
      }
      NI = NI->getFirstPred();
    } while (NI);
  }
  return Ex;
}

bool bugreporter::trackNullOrUndefValue(const ExplodedNode *N,
                                        const Stmt *S,
                                        BugReport &report, bool IsArg,
                                        bool EnableNullFPSuppression) {
  if (!S || !N)
    return false;

  if (const Expr *Ex = dyn_cast<Expr>(S)) {
    Ex = Ex->IgnoreParenCasts();
    const Expr *PeeledEx = peelOffOuterExpr(Ex, N);
    if (Ex != PeeledEx)
      S = PeeledEx;
  }

  const Expr *Inner = nullptr;
  if (const Expr *Ex = dyn_cast<Expr>(S)) {
    Ex = Ex->IgnoreParenCasts();
    if (ExplodedGraph::isInterestingLValueExpr(Ex) || CallEvent::isCallStmt(Ex))
      Inner = Ex;
  }

  if (IsArg && !Inner) {
    assert(N->getLocation().getAs<CallEnter>() && "Tracking arg but not at call");
  } else {
    // Walk through nodes until we get one that matches the statement exactly.
    // Alternately, if we hit a known lvalue for the statement, we know we've
    // gone too far (though we can likely track the lvalue better anyway).
    do {
      const ProgramPoint &pp = N->getLocation();
      if (Optional<StmtPoint> ps = pp.getAs<StmtPoint>()) {
        if (ps->getStmt() == S || ps->getStmt() == Inner)
          break;
      } else if (Optional<CallExitEnd> CEE = pp.getAs<CallExitEnd>()) {
        if (CEE->getCalleeContext()->getCallSite() == S ||
            CEE->getCalleeContext()->getCallSite() == Inner)
          break;
      }
      N = N->getFirstPred();
    } while (N);

    if (!N)
      return false;
  }

  ProgramStateRef state = N->getState();

  // The message send could be nil due to the receiver being nil.
  // At this point in the path, the receiver should be live since we are at the
  // message send expr. If it is nil, start tracking it.
  if (const Expr *Receiver = NilReceiverBRVisitor::getNilReceiver(S, N))
    trackNullOrUndefValue(N, Receiver, report, false, EnableNullFPSuppression);


  // See if the expression we're interested refers to a variable.
  // If so, we can track both its contents and constraints on its value.
  if (Inner && ExplodedGraph::isInterestingLValueExpr(Inner)) {
    const MemRegion *R = nullptr;

    // Find the ExplodedNode where the lvalue (the value of 'Ex')
    // was computed.  We need this for getting the location value.
    const ExplodedNode *LVNode = N;
    while (LVNode) {
      if (Optional<PostStmt> P = LVNode->getLocation().getAs<PostStmt>()) {
        if (P->getStmt() == Inner)
          break;
      }
      LVNode = LVNode->getFirstPred();
    }
    assert(LVNode && "Unable to find the lvalue node.");
    ProgramStateRef LVState = LVNode->getState();
    SVal LVal = LVState->getSVal(Inner, LVNode->getLocationContext());

    if (LVState->isNull(LVal).isConstrainedTrue()) {
      // In case of C++ references, we want to differentiate between a null
      // reference and reference to null pointer.
      // If the LVal is null, check if we are dealing with null reference.
      // For those, we want to track the location of the reference.
      if (const MemRegion *RR = getLocationRegionIfReference(Inner, N))
        R = RR;
    } else {
      R = LVState->getSVal(Inner, LVNode->getLocationContext()).getAsRegion();

      // If this is a C++ reference to a null pointer, we are tracking the
      // pointer. In additon, we should find the store at which the reference
      // got initialized.
      if (const MemRegion *RR = getLocationRegionIfReference(Inner, N)) {
        if (Optional<KnownSVal> KV = LVal.getAs<KnownSVal>())
          report.addVisitor(llvm::make_unique<FindLastStoreBRVisitor>(
              *KV, RR, EnableNullFPSuppression));
      }
    }

    if (R) {
      // Mark both the variable region and its contents as interesting.
      SVal V = LVState->getRawSVal(loc::MemRegionVal(R));

      report.markInteresting(R);
      report.markInteresting(V);
      report.addVisitor(llvm::make_unique<UndefOrNullArgVisitor>(R));

      // If the contents are symbolic, find out when they became null.
      if (V.getAsLocSymbol(/*IncludeBaseRegions*/ true))
        report.addVisitor(llvm::make_unique<TrackConstraintBRVisitor>(
            V.castAs<DefinedSVal>(), false));

      // Add visitor, which will suppress inline defensive checks.
      if (Optional<DefinedSVal> DV = V.getAs<DefinedSVal>()) {
        if (!DV->isZeroConstant() && LVState->isNull(*DV).isConstrainedTrue() &&
            EnableNullFPSuppression) {
          report.addVisitor(
              llvm::make_unique<SuppressInlineDefensiveChecksVisitor>(*DV,
                                                                      LVNode));
        }
      }

      if (Optional<KnownSVal> KV = V.getAs<KnownSVal>())
        report.addVisitor(llvm::make_unique<FindLastStoreBRVisitor>(
            *KV, R, EnableNullFPSuppression));
      return true;
    }
  }

  // If the expression is not an "lvalue expression", we can still
  // track the constraints on its contents.
  SVal V = state->getSValAsScalarOrLoc(S, N->getLocationContext());

  // If the value came from an inlined function call, we should at least make
  // sure that function isn't pruned in our output.
  if (const Expr *E = dyn_cast<Expr>(S))
    S = E->IgnoreParenCasts();

  ReturnVisitor::addVisitorIfNecessary(N, S, report, EnableNullFPSuppression);

  // Uncomment this to find cases where we aren't properly getting the
  // base value that was dereferenced.
  // assert(!V.isUnknownOrUndef());
  // Is it a symbolic value?
  if (Optional<loc::MemRegionVal> L = V.getAs<loc::MemRegionVal>()) {
    // At this point we are dealing with the region's LValue.
    // However, if the rvalue is a symbolic region, we should track it as well.
    // Try to use the correct type when looking up the value.
    SVal RVal;
    if (const Expr *E = dyn_cast<Expr>(S))
      RVal = state->getRawSVal(L.getValue(), E->getType());
    else
      RVal = state->getSVal(L->getRegion());

    const MemRegion *RegionRVal = RVal.getAsRegion();
    report.addVisitor(llvm::make_unique<UndefOrNullArgVisitor>(L->getRegion()));

    if (RegionRVal && isa<SymbolicRegion>(RegionRVal)) {
      report.markInteresting(RegionRVal);
      report.addVisitor(llvm::make_unique<TrackConstraintBRVisitor>(
          loc::MemRegionVal(RegionRVal), false));
    }
  }

  return true;
}

const Expr *NilReceiverBRVisitor::getNilReceiver(const Stmt *S,
                                                 const ExplodedNode *N) {
  const ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(S);
  if (!ME)
    return nullptr;
  if (const Expr *Receiver = ME->getInstanceReceiver()) {
    ProgramStateRef state = N->getState();
    SVal V = state->getSVal(Receiver, N->getLocationContext());
    if (state->isNull(V).isConstrainedTrue())
      return Receiver;
  }
  return nullptr;
}

PathDiagnosticPiece *NilReceiverBRVisitor::VisitNode(const ExplodedNode *N,
                                                     const ExplodedNode *PrevN,
                                                     BugReporterContext &BRC,
                                                     BugReport &BR) {
  Optional<PreStmt> P = N->getLocationAs<PreStmt>();
  if (!P)
    return nullptr;

  const Stmt *S = P->getStmt();
  const Expr *Receiver = getNilReceiver(S, N);
  if (!Receiver)
    return nullptr;

  llvm::SmallString<256> Buf;
  llvm::raw_svector_ostream OS(Buf);

  if (const ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(S)) {
    OS << "'";
    ME->getSelector().print(OS);
    OS << "' not called";
  }
  else {
    OS << "No method is called";
  }
  OS << " because the receiver is nil";

  // The receiver was nil, and hence the method was skipped.
  // Register a BugReporterVisitor to issue a message telling us how
  // the receiver was null.
  bugreporter::trackNullOrUndefValue(N, Receiver, BR, /*IsArg*/ false,
                                     /*EnableNullFPSuppression*/ false);
  // Issue a message saying that the method was skipped.
  PathDiagnosticLocation L(Receiver, BRC.getSourceManager(),
                                     N->getLocationContext());
  return new PathDiagnosticEventPiece(L, OS.str());
}

// Registers every VarDecl inside a Stmt with a last store visitor.
void FindLastStoreBRVisitor::registerStatementVarDecls(BugReport &BR,
                                                const Stmt *S,
                                                bool EnableNullFPSuppression) {
  const ExplodedNode *N = BR.getErrorNode();
  std::deque<const Stmt *> WorkList;
  WorkList.push_back(S);

  while (!WorkList.empty()) {
    const Stmt *Head = WorkList.front();
    WorkList.pop_front();

    ProgramStateRef state = N->getState();
    ProgramStateManager &StateMgr = state->getStateManager();

    if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(Head)) {
      if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
        const VarRegion *R =
        StateMgr.getRegionManager().getVarRegion(VD, N->getLocationContext());

        // What did we load?
        SVal V = state->getSVal(S, N->getLocationContext());

        if (V.getAs<loc::ConcreteInt>() || V.getAs<nonloc::ConcreteInt>()) {
          // Register a new visitor with the BugReport.
          BR.addVisitor(llvm::make_unique<FindLastStoreBRVisitor>(
              V.castAs<KnownSVal>(), R, EnableNullFPSuppression));
        }
      }
    }

    for (const Stmt *SubStmt : Head->children())
      WorkList.push_back(SubStmt);
  }
}

//===----------------------------------------------------------------------===//
// Visitor that tries to report interesting diagnostics from conditions.
//===----------------------------------------------------------------------===//

/// Return the tag associated with this visitor.  This tag will be used
/// to make all PathDiagnosticPieces created by this visitor.
const char *ConditionBRVisitor::getTag() {
  return "ConditionBRVisitor";
}

PathDiagnosticPiece *ConditionBRVisitor::VisitNode(const ExplodedNode *N,
                                                   const ExplodedNode *Prev,
                                                   BugReporterContext &BRC,
                                                   BugReport &BR) {
  PathDiagnosticPiece *piece = VisitNodeImpl(N, Prev, BRC, BR);
  if (piece) {
    piece->setTag(getTag());
    if (PathDiagnosticEventPiece *ev=dyn_cast<PathDiagnosticEventPiece>(piece))
      ev->setPrunable(true, /* override */ false);
  }
  return piece;
}

PathDiagnosticPiece *ConditionBRVisitor::VisitNodeImpl(const ExplodedNode *N,
                                                       const ExplodedNode *Prev,
                                                       BugReporterContext &BRC,
                                                       BugReport &BR) {

  ProgramPoint progPoint = N->getLocation();
  ProgramStateRef CurrentState = N->getState();
  ProgramStateRef PrevState = Prev->getState();

  // Compare the GDMs of the state, because that is where constraints
  // are managed.  Note that ensure that we only look at nodes that
  // were generated by the analyzer engine proper, not checkers.
  if (CurrentState->getGDM().getRoot() ==
      PrevState->getGDM().getRoot())
    return nullptr;

  // If an assumption was made on a branch, it should be caught
  // here by looking at the state transition.
  if (Optional<BlockEdge> BE = progPoint.getAs<BlockEdge>()) {
    const CFGBlock *srcBlk = BE->getSrc();
    if (const Stmt *term = srcBlk->getTerminator())
      return VisitTerminator(term, N, srcBlk, BE->getDst(), BR, BRC);
    return nullptr;
  }

  if (Optional<PostStmt> PS = progPoint.getAs<PostStmt>()) {
    // FIXME: Assuming that BugReporter is a GRBugReporter is a layering
    // violation.
    const std::pair<const ProgramPointTag *, const ProgramPointTag *> &tags =
      cast<GRBugReporter>(BRC.getBugReporter()).
        getEngine().geteagerlyAssumeBinOpBifurcationTags();

    const ProgramPointTag *tag = PS->getTag();
    if (tag == tags.first)
      return VisitTrueTest(cast<Expr>(PS->getStmt()), true,
                           BRC, BR, N);
    if (tag == tags.second)
      return VisitTrueTest(cast<Expr>(PS->getStmt()), false,
                           BRC, BR, N);

    return nullptr;
  }

  return nullptr;
}

PathDiagnosticPiece *
ConditionBRVisitor::VisitTerminator(const Stmt *Term,
                                    const ExplodedNode *N,
                                    const CFGBlock *srcBlk,
                                    const CFGBlock *dstBlk,
                                    BugReport &R,
                                    BugReporterContext &BRC) {
  const Expr *Cond = nullptr;

  switch (Term->getStmtClass()) {
  default:
    return nullptr;
  case Stmt::IfStmtClass:
    Cond = cast<IfStmt>(Term)->getCond();
    break;
  case Stmt::ConditionalOperatorClass:
    Cond = cast<ConditionalOperator>(Term)->getCond();
    break;
  }

  assert(Cond);
  assert(srcBlk->succ_size() == 2);
  const bool tookTrue = *(srcBlk->succ_begin()) == dstBlk;
  return VisitTrueTest(Cond, tookTrue, BRC, R, N);
}

PathDiagnosticPiece *
ConditionBRVisitor::VisitTrueTest(const Expr *Cond,
                                  bool tookTrue,
                                  BugReporterContext &BRC,
                                  BugReport &R,
                                  const ExplodedNode *N) {

  const Expr *Ex = Cond;

  while (true) {
    Ex = Ex->IgnoreParenCasts();
    switch (Ex->getStmtClass()) {
      default:
        return nullptr;
      case Stmt::BinaryOperatorClass:
        return VisitTrueTest(Cond, cast<BinaryOperator>(Ex), tookTrue, BRC,
                             R, N);
      case Stmt::DeclRefExprClass:
        return VisitTrueTest(Cond, cast<DeclRefExpr>(Ex), tookTrue, BRC,
                             R, N);
      case Stmt::UnaryOperatorClass: {
        const UnaryOperator *UO = cast<UnaryOperator>(Ex);
        if (UO->getOpcode() == UO_LNot) {
          tookTrue = !tookTrue;
          Ex = UO->getSubExpr();
          continue;
        }
        return nullptr;
      }
    }
  }
}

bool ConditionBRVisitor::patternMatch(const Expr *Ex, raw_ostream &Out,
                                      BugReporterContext &BRC,
                                      BugReport &report,
                                      const ExplodedNode *N,
                                      Optional<bool> &prunable) {
  const Expr *OriginalExpr = Ex;
  Ex = Ex->IgnoreParenCasts();

  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(Ex)) {
    const bool quotes = isa<VarDecl>(DR->getDecl());
    if (quotes) {
      Out << '\'';
      const LocationContext *LCtx = N->getLocationContext();
      const ProgramState *state = N->getState().get();
      if (const MemRegion *R = state->getLValue(cast<VarDecl>(DR->getDecl()),
                                                LCtx).getAsRegion()) {
        if (report.isInteresting(R))
          prunable = false;
        else {
          const ProgramState *state = N->getState().get();
          SVal V = state->getSVal(R);
          if (report.isInteresting(V))
            prunable = false;
        }
      }
    }
    Out << DR->getDecl()->getDeclName().getAsString();
    if (quotes)
      Out << '\'';
    return quotes;
  }

  if (const IntegerLiteral *IL = dyn_cast<IntegerLiteral>(Ex)) {
    QualType OriginalTy = OriginalExpr->getType();
    if (OriginalTy->isPointerType()) {
      if (IL->getValue() == 0) {
        Out << "null";
        return false;
      }
    }
    else if (OriginalTy->isObjCObjectPointerType()) {
      if (IL->getValue() == 0) {
        Out << "nil";
        return false;
      }
    }

    Out << IL->getValue();
    return false;
  }

  return false;
}

PathDiagnosticPiece *
ConditionBRVisitor::VisitTrueTest(const Expr *Cond,
                                  const BinaryOperator *BExpr,
                                  const bool tookTrue,
                                  BugReporterContext &BRC,
                                  BugReport &R,
                                  const ExplodedNode *N) {

  bool shouldInvert = false;
  Optional<bool> shouldPrune;

  SmallString<128> LhsString, RhsString;
  {
    llvm::raw_svector_ostream OutLHS(LhsString), OutRHS(RhsString);
    const bool isVarLHS = patternMatch(BExpr->getLHS(), OutLHS, BRC, R, N,
                                       shouldPrune);
    const bool isVarRHS = patternMatch(BExpr->getRHS(), OutRHS, BRC, R, N,
                                       shouldPrune);

    shouldInvert = !isVarLHS && isVarRHS;
  }

  BinaryOperator::Opcode Op = BExpr->getOpcode();

  if (BinaryOperator::isAssignmentOp(Op)) {
    // For assignment operators, all that we care about is that the LHS
    // evaluates to "true" or "false".
    return VisitConditionVariable(LhsString, BExpr->getLHS(), tookTrue,
                                  BRC, R, N);
  }

  // For non-assignment operations, we require that we can understand
  // both the LHS and RHS.
  if (LhsString.empty() || RhsString.empty() ||
      !BinaryOperator::isComparisonOp(Op))
    return nullptr;

  // Should we invert the strings if the LHS is not a variable name?
  SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  Out << "Assuming " << (shouldInvert ? RhsString : LhsString) << " is ";

  // Do we need to invert the opcode?
  if (shouldInvert)
    switch (Op) {
      default: break;
      case BO_LT: Op = BO_GT; break;
      case BO_GT: Op = BO_LT; break;
      case BO_LE: Op = BO_GE; break;
      case BO_GE: Op = BO_LE; break;
    }

  if (!tookTrue)
    switch (Op) {
      case BO_EQ: Op = BO_NE; break;
      case BO_NE: Op = BO_EQ; break;
      case BO_LT: Op = BO_GE; break;
      case BO_GT: Op = BO_LE; break;
      case BO_LE: Op = BO_GT; break;
      case BO_GE: Op = BO_LT; break;
      default:
        return nullptr;
    }

  switch (Op) {
    case BO_EQ:
      Out << "equal to ";
      break;
    case BO_NE:
      Out << "not equal to ";
      break;
    default:
      Out << BinaryOperator::getOpcodeStr(Op) << ' ';
      break;
  }

  Out << (shouldInvert ? LhsString : RhsString);
  const LocationContext *LCtx = N->getLocationContext();
  PathDiagnosticLocation Loc(Cond, BRC.getSourceManager(), LCtx);
  PathDiagnosticEventPiece *event =
    new PathDiagnosticEventPiece(Loc, Out.str());
  if (shouldPrune.hasValue())
    event->setPrunable(shouldPrune.getValue());
  return event;
}

PathDiagnosticPiece *
ConditionBRVisitor::VisitConditionVariable(StringRef LhsString,
                                           const Expr *CondVarExpr,
                                           const bool tookTrue,
                                           BugReporterContext &BRC,
                                           BugReport &report,
                                           const ExplodedNode *N) {
  // FIXME: If there's already a constraint tracker for this variable,
  // we shouldn't emit anything here (c.f. the double note in
  // test/Analysis/inlining/path-notes.c)
  SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  Out << "Assuming " << LhsString << " is ";

  QualType Ty = CondVarExpr->getType();

  if (Ty->isPointerType())
    Out << (tookTrue ? "not null" : "null");
  else if (Ty->isObjCObjectPointerType())
    Out << (tookTrue ? "not nil" : "nil");
  else if (Ty->isBooleanType())
    Out << (tookTrue ? "true" : "false");
  else if (Ty->isIntegralOrEnumerationType())
    Out << (tookTrue ? "non-zero" : "zero");
  else
    return nullptr;

  const LocationContext *LCtx = N->getLocationContext();
  PathDiagnosticLocation Loc(CondVarExpr, BRC.getSourceManager(), LCtx);
  PathDiagnosticEventPiece *event =
    new PathDiagnosticEventPiece(Loc, Out.str());

  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(CondVarExpr)) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
      const ProgramState *state = N->getState().get();
      if (const MemRegion *R = state->getLValue(VD, LCtx).getAsRegion()) {
        if (report.isInteresting(R))
          event->setPrunable(false);
      }
    }
  }

  return event;
}

PathDiagnosticPiece *
ConditionBRVisitor::VisitTrueTest(const Expr *Cond,
                                  const DeclRefExpr *DR,
                                  const bool tookTrue,
                                  BugReporterContext &BRC,
                                  BugReport &report,
                                  const ExplodedNode *N) {

  const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl());
  if (!VD)
    return nullptr;

  SmallString<256> Buf;
  llvm::raw_svector_ostream Out(Buf);

  Out << "Assuming '" << VD->getDeclName() << "' is ";

  QualType VDTy = VD->getType();

  if (VDTy->isPointerType())
    Out << (tookTrue ? "non-null" : "null");
  else if (VDTy->isObjCObjectPointerType())
    Out << (tookTrue ? "non-nil" : "nil");
  else if (VDTy->isScalarType())
    Out << (tookTrue ? "not equal to 0" : "0");
  else
    return nullptr;

  const LocationContext *LCtx = N->getLocationContext();
  PathDiagnosticLocation Loc(Cond, BRC.getSourceManager(), LCtx);
  PathDiagnosticEventPiece *event =
    new PathDiagnosticEventPiece(Loc, Out.str());

  const ProgramState *state = N->getState().get();
  if (const MemRegion *R = state->getLValue(VD, LCtx).getAsRegion()) {
    if (report.isInteresting(R))
      event->setPrunable(false);
    else {
      SVal V = state->getSVal(R);
      if (report.isInteresting(V))
        event->setPrunable(false);
    }
  }
  return event;
}


// FIXME: Copied from ExprEngineCallAndReturn.cpp.
static bool isInStdNamespace(const Decl *D) {
  const DeclContext *DC = D->getDeclContext()->getEnclosingNamespaceContext();
  const NamespaceDecl *ND = dyn_cast<NamespaceDecl>(DC);
  if (!ND)
    return false;

  while (const NamespaceDecl *Parent = dyn_cast<NamespaceDecl>(ND->getParent()))
    ND = Parent;

  return ND->isStdNamespace();
}

std::unique_ptr<PathDiagnosticPiece>
LikelyFalsePositiveSuppressionBRVisitor::getEndPath(BugReporterContext &BRC,
                                                    const ExplodedNode *N,
                                                    BugReport &BR) {
  // Here we suppress false positives coming from system headers. This list is
  // based on known issues.
  ExprEngine &Eng = BRC.getBugReporter().getEngine();
  AnalyzerOptions &Options = Eng.getAnalysisManager().options;
  const Decl *D = N->getLocationContext()->getDecl();

  if (isInStdNamespace(D)) {
    // Skip reports within the 'std' namespace. Although these can sometimes be
    // the user's fault, we currently don't report them very well, and
    // Note that this will not help for any other data structure libraries, like
    // TR1, Boost, or llvm/ADT.
    if (Options.shouldSuppressFromCXXStandardLibrary()) {
      BR.markInvalid(getTag(), nullptr);
      return nullptr;

    } else {
      // If the complete 'std' suppression is not enabled, suppress reports
      // from the 'std' namespace that are known to produce false positives.

      // The analyzer issues a false use-after-free when std::list::pop_front
      // or std::list::pop_back are called multiple times because we cannot
      // reason about the internal invariants of the datastructure.
      if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(D)) {
        const CXXRecordDecl *CD = MD->getParent();
        if (CD->getName() == "list") {
          BR.markInvalid(getTag(), nullptr);
          return nullptr;
        }
      }

      // The analyzer issues a false positive when the constructor of
      // std::__independent_bits_engine from algorithms is used.
      if (const CXXConstructorDecl *MD = dyn_cast<CXXConstructorDecl>(D)) {
        const CXXRecordDecl *CD = MD->getParent();
        if (CD->getName() == "__independent_bits_engine") {
          BR.markInvalid(getTag(), nullptr);
          return nullptr;
        }
      }

      // The analyzer issues a false positive on
      //   std::basic_string<uint8_t> v; v.push_back(1);
      // and
      //   std::u16string s; s += u'a';
      // because we cannot reason about the internal invariants of the
      // datastructure.
      for (const LocationContext *LCtx = N->getLocationContext(); LCtx;
           LCtx = LCtx->getParent()) {
        const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(LCtx->getDecl());
        if (!MD)
          continue;

        const CXXRecordDecl *CD = MD->getParent();
        if (CD->getName() == "basic_string") {
          BR.markInvalid(getTag(), nullptr);
          return nullptr;
        }
      }
    }
  }

  // Skip reports within the sys/queue.h macros as we do not have the ability to
  // reason about data structure shapes.
  SourceManager &SM = BRC.getSourceManager();
  FullSourceLoc Loc = BR.getLocation(SM).asLocation();
  while (Loc.isMacroID()) {
    Loc = Loc.getSpellingLoc();
    if (SM.getFilename(Loc).endswith("sys/queue.h")) {
      BR.markInvalid(getTag(), nullptr);
      return nullptr;
    }
  }

  return nullptr;
}

PathDiagnosticPiece *
UndefOrNullArgVisitor::VisitNode(const ExplodedNode *N,
                                  const ExplodedNode *PrevN,
                                  BugReporterContext &BRC,
                                  BugReport &BR) {

  ProgramStateRef State = N->getState();
  ProgramPoint ProgLoc = N->getLocation();

  // We are only interested in visiting CallEnter nodes.
  Optional<CallEnter> CEnter = ProgLoc.getAs<CallEnter>();
  if (!CEnter)
    return nullptr;

  // Check if one of the arguments is the region the visitor is tracking.
  CallEventManager &CEMgr = BRC.getStateManager().getCallEventManager();
  CallEventRef<> Call = CEMgr.getCaller(CEnter->getCalleeContext(), State);
  unsigned Idx = 0;
  ArrayRef<ParmVarDecl*> parms = Call->parameters();

  for (ArrayRef<ParmVarDecl*>::iterator I = parms.begin(), E = parms.end();
                              I != E; ++I, ++Idx) {
    const MemRegion *ArgReg = Call->getArgSVal(Idx).getAsRegion();

    // Are we tracking the argument or its subregion?
    if ( !ArgReg || (ArgReg != R && !R->isSubRegionOf(ArgReg->StripCasts())))
      continue;

    // Check the function parameter type.
    const ParmVarDecl *ParamDecl = *I;
    assert(ParamDecl && "Formal parameter has no decl?");
    QualType T = ParamDecl->getType();

    if (!(T->isAnyPointerType() || T->isReferenceType())) {
      // Function can only change the value passed in by address.
      continue;
    }

    // If it is a const pointer value, the function does not intend to
    // change the value.
    if (T->getPointeeType().isConstQualified())
      continue;

    // Mark the call site (LocationContext) as interesting if the value of the
    // argument is undefined or '0'/'NULL'.
    SVal BoundVal = State->getSVal(R);
    if (BoundVal.isUndef() || BoundVal.isZeroConstant()) {
      BR.markInteresting(CEnter->getCalleeContext());
      return nullptr;
    }
  }
  return nullptr;
}
