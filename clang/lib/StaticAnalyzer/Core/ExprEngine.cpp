//=-- ExprEngine.cpp - Path-Sensitive Expression-Level Dataflow ---*- C++ -*-=
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a meta-engine for path-sensitive dataflow analysis that
//  is built on GREngine, but provides the boilerplate to execute transfer
//  functions and build the ExplodedGraph at the expression level.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "PrettyStackTraceLocationContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/Basic/SourceManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/LoopWidening.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"

#ifndef NDEBUG
#include "llvm/Support/GraphWriter.h"
#endif

using namespace clang;
using namespace ento;
using llvm::APSInt;

#define DEBUG_TYPE "ExprEngine"

STATISTIC(NumRemoveDeadBindings,
            "The # of times RemoveDeadBindings is called");
STATISTIC(NumMaxBlockCountReached,
            "The # of aborted paths due to reaching the maximum block count in "
            "a top level function");
STATISTIC(NumMaxBlockCountReachedInInlined,
            "The # of aborted paths due to reaching the maximum block count in "
            "an inlined function");
STATISTIC(NumTimesRetriedWithoutInlining,
            "The # of times we re-evaluated a call without inlining");

typedef std::pair<const CXXBindTemporaryExpr *, const StackFrameContext *>
    CXXBindTemporaryContext;

// Keeps track of whether CXXBindTemporaryExpr nodes have been evaluated.
// The StackFrameContext assures that nested calls due to inlined recursive
// functions do not interfere.
REGISTER_TRAIT_WITH_PROGRAMSTATE(InitializedTemporariesSet,
                                 llvm::ImmutableSet<CXXBindTemporaryContext>)

//===----------------------------------------------------------------------===//
// Engine construction and deletion.
//===----------------------------------------------------------------------===//

static const char* TagProviderName = "ExprEngine";

ExprEngine::ExprEngine(AnalysisManager &mgr, bool gcEnabled,
                       SetOfConstDecls *VisitedCalleesIn,
                       FunctionSummariesTy *FS,
                       InliningModes HowToInlineIn)
  : AMgr(mgr),
    AnalysisDeclContexts(mgr.getAnalysisDeclContextManager()),
    Engine(*this, FS),
    G(Engine.getGraph()),
    StateMgr(getContext(), mgr.getStoreManagerCreator(),
             mgr.getConstraintManagerCreator(), G.getAllocator(),
             this),
    SymMgr(StateMgr.getSymbolManager()),
    svalBuilder(StateMgr.getSValBuilder()),
    currStmtIdx(0), currBldrCtx(nullptr),
    ObjCNoRet(mgr.getASTContext()),
    ObjCGCEnabled(gcEnabled), BR(mgr, *this),
    VisitedCallees(VisitedCalleesIn),
    HowToInline(HowToInlineIn)
{
  unsigned TrimInterval = mgr.options.getGraphTrimInterval();
  if (TrimInterval != 0) {
    // Enable eager node reclaimation when constructing the ExplodedGraph.
    G.enableNodeReclamation(TrimInterval);
  }
}

ExprEngine::~ExprEngine() {
  BR.FlushReports();
}

//===----------------------------------------------------------------------===//
// Utility methods.
//===----------------------------------------------------------------------===//

ProgramStateRef ExprEngine::getInitialState(const LocationContext *InitLoc) {
  ProgramStateRef state = StateMgr.getInitialState(InitLoc);
  const Decl *D = InitLoc->getDecl();

  // Preconditions.
  // FIXME: It would be nice if we had a more general mechanism to add
  // such preconditions.  Some day.
  do {

    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      // Precondition: the first argument of 'main' is an integer guaranteed
      //  to be > 0.
      const IdentifierInfo *II = FD->getIdentifier();
      if (!II || !(II->getName() == "main" && FD->getNumParams() > 0))
        break;

      const ParmVarDecl *PD = FD->getParamDecl(0);
      QualType T = PD->getType();
      const BuiltinType *BT = dyn_cast<BuiltinType>(T);
      if (!BT || !BT->isInteger())
        break;

      const MemRegion *R = state->getRegion(PD, InitLoc);
      if (!R)
        break;

      SVal V = state->getSVal(loc::MemRegionVal(R));
      SVal Constraint_untested = evalBinOp(state, BO_GT, V,
                                           svalBuilder.makeZeroVal(T),
                                           svalBuilder.getConditionType());

      Optional<DefinedOrUnknownSVal> Constraint =
          Constraint_untested.getAs<DefinedOrUnknownSVal>();

      if (!Constraint)
        break;

      if (ProgramStateRef newState = state->assume(*Constraint, true))
        state = newState;
    }
    break;
  }
  while (0);

  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    // Precondition: 'self' is always non-null upon entry to an Objective-C
    // method.
    const ImplicitParamDecl *SelfD = MD->getSelfDecl();
    const MemRegion *R = state->getRegion(SelfD, InitLoc);
    SVal V = state->getSVal(loc::MemRegionVal(R));

    if (Optional<Loc> LV = V.getAs<Loc>()) {
      // Assume that the pointer value in 'self' is non-null.
      state = state->assume(*LV, true);
      assert(state && "'self' cannot be null");
    }
  }

  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(D)) {
    if (!MD->isStatic()) {
      // Precondition: 'this' is always non-null upon entry to the
      // top-level function.  This is our starting assumption for
      // analyzing an "open" program.
      const StackFrameContext *SFC = InitLoc->getCurrentStackFrame();
      if (SFC->getParent() == nullptr) {
        loc::MemRegionVal L = svalBuilder.getCXXThis(MD, SFC);
        SVal V = state->getSVal(L);
        if (Optional<Loc> LV = V.getAs<Loc>()) {
          state = state->assume(*LV, true);
          assert(state && "'this' cannot be null");
        }
      }
    }
  }

  return state;
}

ProgramStateRef
ExprEngine::createTemporaryRegionIfNeeded(ProgramStateRef State,
                                          const LocationContext *LC,
                                          const Expr *Ex,
                                          const Expr *Result) {
  SVal V = State->getSVal(Ex, LC);
  if (!Result) {
    // If we don't have an explicit result expression, we're in "if needed"
    // mode. Only create a region if the current value is a NonLoc.
    if (!V.getAs<NonLoc>())
      return State;
    Result = Ex;
  } else {
    // We need to create a region no matter what. For sanity, make sure we don't
    // try to stuff a Loc into a non-pointer temporary region.
    assert(!V.getAs<Loc>() || Loc::isLocType(Result->getType()) ||
           Result->getType()->isMemberPointerType());
  }

  ProgramStateManager &StateMgr = State->getStateManager();
  MemRegionManager &MRMgr = StateMgr.getRegionManager();
  StoreManager &StoreMgr = StateMgr.getStoreManager();

  // We need to be careful about treating a derived type's value as
  // bindings for a base type. Unless we're creating a temporary pointer region,
  // start by stripping and recording base casts.
  SmallVector<const CastExpr *, 4> Casts;
  const Expr *Inner = Ex->IgnoreParens();
  if (!Loc::isLocType(Result->getType())) {
    while (const CastExpr *CE = dyn_cast<CastExpr>(Inner)) {
      if (CE->getCastKind() == CK_DerivedToBase ||
          CE->getCastKind() == CK_UncheckedDerivedToBase)
        Casts.push_back(CE);
      else if (CE->getCastKind() != CK_NoOp)
        break;

      Inner = CE->getSubExpr()->IgnoreParens();
    }
  }

  // Create a temporary object region for the inner expression (which may have
  // a more derived type) and bind the value into it.
  const TypedValueRegion *TR = nullptr;
  if (const MaterializeTemporaryExpr *MT =
          dyn_cast<MaterializeTemporaryExpr>(Result)) {
    StorageDuration SD = MT->getStorageDuration();
    // If this object is bound to a reference with static storage duration, we
    // put it in a different region to prevent "address leakage" warnings.
    if (SD == SD_Static || SD == SD_Thread)
        TR = MRMgr.getCXXStaticTempObjectRegion(Inner);
  }
  if (!TR)
    TR = MRMgr.getCXXTempObjectRegion(Inner, LC);

  SVal Reg = loc::MemRegionVal(TR);

  if (V.isUnknown())
    V = getSValBuilder().conjureSymbolVal(Result, LC, TR->getValueType(),
                                          currBldrCtx->blockCount());
  State = State->bindLoc(Reg, V);

  // Re-apply the casts (from innermost to outermost) for type sanity.
  for (SmallVectorImpl<const CastExpr *>::reverse_iterator I = Casts.rbegin(),
                                                           E = Casts.rend();
       I != E; ++I) {
    Reg = StoreMgr.evalDerivedToBase(Reg, *I);
  }

  State = State->BindExpr(Result, LC, Reg);
  return State;
}

//===----------------------------------------------------------------------===//
// Top-level transfer function logic (Dispatcher).
//===----------------------------------------------------------------------===//

/// evalAssume - Called by ConstraintManager. Used to call checker-specific
///  logic for handling assumptions on symbolic values.
ProgramStateRef ExprEngine::processAssume(ProgramStateRef state,
                                              SVal cond, bool assumption) {
  return getCheckerManager().runCheckersForEvalAssume(state, cond, assumption);
}

bool ExprEngine::wantsRegionChangeUpdate(ProgramStateRef state) {
  return getCheckerManager().wantsRegionChangeUpdate(state);
}

ProgramStateRef
ExprEngine::processRegionChanges(ProgramStateRef state,
                                 const InvalidatedSymbols *invalidated,
                                 ArrayRef<const MemRegion *> Explicits,
                                 ArrayRef<const MemRegion *> Regions,
                                 const CallEvent *Call) {
  return getCheckerManager().runCheckersForRegionChanges(state, invalidated,
                                                      Explicits, Regions, Call);
}

void ExprEngine::printState(raw_ostream &Out, ProgramStateRef State,
                            const char *NL, const char *Sep) {
  getCheckerManager().runCheckersForPrintState(Out, State, NL, Sep);
}

void ExprEngine::processEndWorklist(bool hasWorkRemaining) {
  getCheckerManager().runCheckersForEndAnalysis(G, BR, *this);
}

void ExprEngine::processCFGElement(const CFGElement E, ExplodedNode *Pred,
                                   unsigned StmtIdx, NodeBuilderContext *Ctx) {
  PrettyStackTraceLocationContext CrashInfo(Pred->getLocationContext());
  currStmtIdx = StmtIdx;
  currBldrCtx = Ctx;

  switch (E.getKind()) {
    case CFGElement::Statement:
      ProcessStmt(const_cast<Stmt*>(E.castAs<CFGStmt>().getStmt()), Pred);
      return;
    case CFGElement::Initializer:
      ProcessInitializer(E.castAs<CFGInitializer>().getInitializer(), Pred);
      return;
    case CFGElement::NewAllocator:
      ProcessNewAllocator(E.castAs<CFGNewAllocator>().getAllocatorExpr(),
                          Pred);
      return;
    case CFGElement::AutomaticObjectDtor:
    case CFGElement::DeleteDtor:
    case CFGElement::BaseDtor:
    case CFGElement::MemberDtor:
    case CFGElement::TemporaryDtor:
      ProcessImplicitDtor(E.castAs<CFGImplicitDtor>(), Pred);
      return;
  }
}

static bool shouldRemoveDeadBindings(AnalysisManager &AMgr,
                                     const CFGStmt S,
                                     const ExplodedNode *Pred,
                                     const LocationContext *LC) {

  // Are we never purging state values?
  if (AMgr.options.AnalysisPurgeOpt == PurgeNone)
    return false;

  // Is this the beginning of a basic block?
  if (Pred->getLocation().getAs<BlockEntrance>())
    return true;

  // Is this on a non-expression?
  if (!isa<Expr>(S.getStmt()))
    return true;

  // Run before processing a call.
  if (CallEvent::isCallStmt(S.getStmt()))
    return true;

  // Is this an expression that is consumed by another expression?  If so,
  // postpone cleaning out the state.
  ParentMap &PM = LC->getAnalysisDeclContext()->getParentMap();
  return !PM.isConsumedExpr(cast<Expr>(S.getStmt()));
}

void ExprEngine::removeDead(ExplodedNode *Pred, ExplodedNodeSet &Out,
                            const Stmt *ReferenceStmt,
                            const LocationContext *LC,
                            const Stmt *DiagnosticStmt,
                            ProgramPoint::Kind K) {
  assert((K == ProgramPoint::PreStmtPurgeDeadSymbolsKind ||
          ReferenceStmt == nullptr || isa<ReturnStmt>(ReferenceStmt))
          && "PostStmt is not generally supported by the SymbolReaper yet");
  assert(LC && "Must pass the current (or expiring) LocationContext");

  if (!DiagnosticStmt) {
    DiagnosticStmt = ReferenceStmt;
    assert(DiagnosticStmt && "Required for clearing a LocationContext");
  }

  NumRemoveDeadBindings++;
  ProgramStateRef CleanedState = Pred->getState();

  // LC is the location context being destroyed, but SymbolReaper wants a
  // location context that is still live. (If this is the top-level stack
  // frame, this will be null.)
  if (!ReferenceStmt) {
    assert(K == ProgramPoint::PostStmtPurgeDeadSymbolsKind &&
           "Use PostStmtPurgeDeadSymbolsKind for clearing a LocationContext");
    LC = LC->getParent();
  }

  const StackFrameContext *SFC = LC ? LC->getCurrentStackFrame() : nullptr;
  SymbolReaper SymReaper(SFC, ReferenceStmt, SymMgr, getStoreManager());

  getCheckerManager().runCheckersForLiveSymbols(CleanedState, SymReaper);

  // Create a state in which dead bindings are removed from the environment
  // and the store. TODO: The function should just return new env and store,
  // not a new state.
  CleanedState = StateMgr.removeDeadBindings(CleanedState, SFC, SymReaper);

  // Process any special transfer function for dead symbols.
  // A tag to track convenience transitions, which can be removed at cleanup.
  static SimpleProgramPointTag cleanupTag(TagProviderName, "Clean Node");
  if (!SymReaper.hasDeadSymbols()) {
    // Generate a CleanedNode that has the environment and store cleaned
    // up. Since no symbols are dead, we can optimize and not clean out
    // the constraint manager.
    StmtNodeBuilder Bldr(Pred, Out, *currBldrCtx);
    Bldr.generateNode(DiagnosticStmt, Pred, CleanedState, &cleanupTag, K);

  } else {
    // Call checkers with the non-cleaned state so that they could query the
    // values of the soon to be dead symbols.
    ExplodedNodeSet CheckedSet;
    getCheckerManager().runCheckersForDeadSymbols(CheckedSet, Pred, SymReaper,
                                                  DiagnosticStmt, *this, K);

    // For each node in CheckedSet, generate CleanedNodes that have the
    // environment, the store, and the constraints cleaned up but have the
    // user-supplied states as the predecessors.
    StmtNodeBuilder Bldr(CheckedSet, Out, *currBldrCtx);
    for (ExplodedNodeSet::const_iterator
          I = CheckedSet.begin(), E = CheckedSet.end(); I != E; ++I) {
      ProgramStateRef CheckerState = (*I)->getState();

      // The constraint manager has not been cleaned up yet, so clean up now.
      CheckerState = getConstraintManager().removeDeadBindings(CheckerState,
                                                               SymReaper);

      assert(StateMgr.haveEqualEnvironments(CheckerState, Pred->getState()) &&
        "Checkers are not allowed to modify the Environment as a part of "
        "checkDeadSymbols processing.");
      assert(StateMgr.haveEqualStores(CheckerState, Pred->getState()) &&
        "Checkers are not allowed to modify the Store as a part of "
        "checkDeadSymbols processing.");

      // Create a state based on CleanedState with CheckerState GDM and
      // generate a transition to that state.
      ProgramStateRef CleanedCheckerSt =
        StateMgr.getPersistentStateWithGDM(CleanedState, CheckerState);
      Bldr.generateNode(DiagnosticStmt, *I, CleanedCheckerSt, &cleanupTag, K);
    }
  }
}

void ExprEngine::ProcessStmt(const CFGStmt S,
                             ExplodedNode *Pred) {
  // Reclaim any unnecessary nodes in the ExplodedGraph.
  G.reclaimRecentlyAllocatedNodes();

  const Stmt *currStmt = S.getStmt();
  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                currStmt->getLocStart(),
                                "Error evaluating statement");

  // Remove dead bindings and symbols.
  ExplodedNodeSet CleanedStates;
  if (shouldRemoveDeadBindings(AMgr, S, Pred, Pred->getLocationContext())){
    removeDead(Pred, CleanedStates, currStmt, Pred->getLocationContext());
  } else
    CleanedStates.Add(Pred);

  // Visit the statement.
  ExplodedNodeSet Dst;
  for (ExplodedNodeSet::iterator I = CleanedStates.begin(),
                                 E = CleanedStates.end(); I != E; ++I) {
    ExplodedNodeSet DstI;
    // Visit the statement.
    Visit(currStmt, *I, DstI);
    Dst.insert(DstI);
  }

  // Enqueue the new nodes onto the work list.
  Engine.enqueue(Dst, currBldrCtx->getBlock(), currStmtIdx);
}

void ExprEngine::ProcessInitializer(const CFGInitializer Init,
                                    ExplodedNode *Pred) {
  const CXXCtorInitializer *BMI = Init.getInitializer();

  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                BMI->getSourceLocation(),
                                "Error evaluating initializer");

  // We don't clean up dead bindings here.
  const StackFrameContext *stackFrame =
                           cast<StackFrameContext>(Pred->getLocationContext());
  const CXXConstructorDecl *decl =
                           cast<CXXConstructorDecl>(stackFrame->getDecl());

  ProgramStateRef State = Pred->getState();
  SVal thisVal = State->getSVal(svalBuilder.getCXXThis(decl, stackFrame));

  ExplodedNodeSet Tmp(Pred);
  SVal FieldLoc;

  // Evaluate the initializer, if necessary
  if (BMI->isAnyMemberInitializer()) {
    // Constructors build the object directly in the field,
    // but non-objects must be copied in from the initializer.
    if (auto *CtorExpr = findDirectConstructorForCurrentCFGElement()) {
      assert(BMI->getInit()->IgnoreImplicit() == CtorExpr);
      (void)CtorExpr;
      // The field was directly constructed, so there is no need to bind.
    } else {
      const Expr *Init = BMI->getInit()->IgnoreImplicit();
      const ValueDecl *Field;
      if (BMI->isIndirectMemberInitializer()) {
        Field = BMI->getIndirectMember();
        FieldLoc = State->getLValue(BMI->getIndirectMember(), thisVal);
      } else {
        Field = BMI->getMember();
        FieldLoc = State->getLValue(BMI->getMember(), thisVal);
      }

      SVal InitVal;
      if (BMI->getNumArrayIndices() > 0) {
        // Handle arrays of trivial type. We can represent this with a
        // primitive load/copy from the base array region.
        const ArraySubscriptExpr *ASE;
        while ((ASE = dyn_cast<ArraySubscriptExpr>(Init)))
          Init = ASE->getBase()->IgnoreImplicit();

        SVal LValue = State->getSVal(Init, stackFrame);
        if (Optional<Loc> LValueLoc = LValue.getAs<Loc>())
          InitVal = State->getSVal(*LValueLoc);

        // If we fail to get the value for some reason, use a symbolic value.
        if (InitVal.isUnknownOrUndef()) {
          SValBuilder &SVB = getSValBuilder();
          InitVal = SVB.conjureSymbolVal(BMI->getInit(), stackFrame,
                                         Field->getType(),
                                         currBldrCtx->blockCount());
        }
      } else {
        InitVal = State->getSVal(BMI->getInit(), stackFrame);
      }

      assert(Tmp.size() == 1 && "have not generated any new nodes yet");
      assert(*Tmp.begin() == Pred && "have not generated any new nodes yet");
      Tmp.clear();

      PostInitializer PP(BMI, FieldLoc.getAsRegion(), stackFrame);
      evalBind(Tmp, Init, Pred, FieldLoc, InitVal, /*isInit=*/true, &PP);
    }
  } else {
    assert(BMI->isBaseInitializer() || BMI->isDelegatingInitializer());
    // We already did all the work when visiting the CXXConstructExpr.
  }

  // Construct PostInitializer nodes whether the state changed or not,
  // so that the diagnostics don't get confused.
  PostInitializer PP(BMI, FieldLoc.getAsRegion(), stackFrame);
  ExplodedNodeSet Dst;
  NodeBuilder Bldr(Tmp, Dst, *currBldrCtx);
  for (ExplodedNodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I != E; ++I) {
    ExplodedNode *N = *I;
    Bldr.generateNode(PP, N->getState(), N);
  }

  // Enqueue the new nodes onto the work list.
  Engine.enqueue(Dst, currBldrCtx->getBlock(), currStmtIdx);
}

void ExprEngine::ProcessImplicitDtor(const CFGImplicitDtor D,
                                     ExplodedNode *Pred) {
  ExplodedNodeSet Dst;
  switch (D.getKind()) {
  case CFGElement::AutomaticObjectDtor:
    ProcessAutomaticObjDtor(D.castAs<CFGAutomaticObjDtor>(), Pred, Dst);
    break;
  case CFGElement::BaseDtor:
    ProcessBaseDtor(D.castAs<CFGBaseDtor>(), Pred, Dst);
    break;
  case CFGElement::MemberDtor:
    ProcessMemberDtor(D.castAs<CFGMemberDtor>(), Pred, Dst);
    break;
  case CFGElement::TemporaryDtor:
    ProcessTemporaryDtor(D.castAs<CFGTemporaryDtor>(), Pred, Dst);
    break;
  case CFGElement::DeleteDtor:
    ProcessDeleteDtor(D.castAs<CFGDeleteDtor>(), Pred, Dst);
    break;
  default:
    llvm_unreachable("Unexpected dtor kind.");
  }

  // Enqueue the new nodes onto the work list.
  Engine.enqueue(Dst, currBldrCtx->getBlock(), currStmtIdx);
}

void ExprEngine::ProcessNewAllocator(const CXXNewExpr *NE,
                                     ExplodedNode *Pred) {
  ExplodedNodeSet Dst;
  AnalysisManager &AMgr = getAnalysisManager();
  AnalyzerOptions &Opts = AMgr.options;
  // TODO: We're not evaluating allocators for all cases just yet as
  // we're not handling the return value correctly, which causes false
  // positives when the alpha.cplusplus.NewDeleteLeaks check is on.
  if (Opts.mayInlineCXXAllocator())
    VisitCXXNewAllocatorCall(NE, Pred, Dst);
  else {
    NodeBuilder Bldr(Pred, Dst, *currBldrCtx);
    const LocationContext *LCtx = Pred->getLocationContext();
    PostImplicitCall PP(NE->getOperatorNew(), NE->getLocStart(), LCtx);
    Bldr.generateNode(PP, Pred->getState(), Pred);
  }
  Engine.enqueue(Dst, currBldrCtx->getBlock(), currStmtIdx);
}

void ExprEngine::ProcessAutomaticObjDtor(const CFGAutomaticObjDtor Dtor,
                                         ExplodedNode *Pred,
                                         ExplodedNodeSet &Dst) {
  const VarDecl *varDecl = Dtor.getVarDecl();
  QualType varType = varDecl->getType();

  ProgramStateRef state = Pred->getState();
  SVal dest = state->getLValue(varDecl, Pred->getLocationContext());
  const MemRegion *Region = dest.castAs<loc::MemRegionVal>().getRegion();

  if (const ReferenceType *refType = varType->getAs<ReferenceType>()) {
    varType = refType->getPointeeType();
    Region = state->getSVal(Region).getAsRegion();
  }

  VisitCXXDestructor(varType, Region, Dtor.getTriggerStmt(), /*IsBase=*/ false,
                     Pred, Dst);
}

void ExprEngine::ProcessDeleteDtor(const CFGDeleteDtor Dtor,
                                   ExplodedNode *Pred,
                                   ExplodedNodeSet &Dst) {
  ProgramStateRef State = Pred->getState();
  const LocationContext *LCtx = Pred->getLocationContext();
  const CXXDeleteExpr *DE = Dtor.getDeleteExpr();
  const Stmt *Arg = DE->getArgument();
  SVal ArgVal = State->getSVal(Arg, LCtx);

  // If the argument to delete is known to be a null value,
  // don't run destructor.
  if (State->isNull(ArgVal).isConstrainedTrue()) {
    QualType DTy = DE->getDestroyedType();
    QualType BTy = getContext().getBaseElementType(DTy);
    const CXXRecordDecl *RD = BTy->getAsCXXRecordDecl();
    const CXXDestructorDecl *Dtor = RD->getDestructor();

    PostImplicitCall PP(Dtor, DE->getLocStart(), LCtx);
    NodeBuilder Bldr(Pred, Dst, *currBldrCtx);
    Bldr.generateNode(PP, Pred->getState(), Pred);
    return;
  }

  VisitCXXDestructor(DE->getDestroyedType(),
                     ArgVal.getAsRegion(),
                     DE, /*IsBase=*/ false,
                     Pred, Dst);
}

void ExprEngine::ProcessBaseDtor(const CFGBaseDtor D,
                                 ExplodedNode *Pred, ExplodedNodeSet &Dst) {
  const LocationContext *LCtx = Pred->getLocationContext();

  const CXXDestructorDecl *CurDtor = cast<CXXDestructorDecl>(LCtx->getDecl());
  Loc ThisPtr = getSValBuilder().getCXXThis(CurDtor,
                                            LCtx->getCurrentStackFrame());
  SVal ThisVal = Pred->getState()->getSVal(ThisPtr);

  // Create the base object region.
  const CXXBaseSpecifier *Base = D.getBaseSpecifier();
  QualType BaseTy = Base->getType();
  SVal BaseVal = getStoreManager().evalDerivedToBase(ThisVal, BaseTy,
                                                     Base->isVirtual());

  VisitCXXDestructor(BaseTy, BaseVal.castAs<loc::MemRegionVal>().getRegion(),
                     CurDtor->getBody(), /*IsBase=*/ true, Pred, Dst);
}

void ExprEngine::ProcessMemberDtor(const CFGMemberDtor D,
                                   ExplodedNode *Pred, ExplodedNodeSet &Dst) {
  const FieldDecl *Member = D.getFieldDecl();
  ProgramStateRef State = Pred->getState();
  const LocationContext *LCtx = Pred->getLocationContext();

  const CXXDestructorDecl *CurDtor = cast<CXXDestructorDecl>(LCtx->getDecl());
  Loc ThisVal = getSValBuilder().getCXXThis(CurDtor,
                                            LCtx->getCurrentStackFrame());
  SVal FieldVal =
      State->getLValue(Member, State->getSVal(ThisVal).castAs<Loc>());

  VisitCXXDestructor(Member->getType(),
                     FieldVal.castAs<loc::MemRegionVal>().getRegion(),
                     CurDtor->getBody(), /*IsBase=*/false, Pred, Dst);
}

void ExprEngine::ProcessTemporaryDtor(const CFGTemporaryDtor D,
                                      ExplodedNode *Pred,
                                      ExplodedNodeSet &Dst) {
  ExplodedNodeSet CleanDtorState;
  StmtNodeBuilder StmtBldr(Pred, CleanDtorState, *currBldrCtx);
  ProgramStateRef State = Pred->getState();
  if (State->contains<InitializedTemporariesSet>(
      std::make_pair(D.getBindTemporaryExpr(), Pred->getStackFrame()))) {
    // FIXME: Currently we insert temporary destructors for default parameters,
    // but we don't insert the constructors.
    State = State->remove<InitializedTemporariesSet>(
        std::make_pair(D.getBindTemporaryExpr(), Pred->getStackFrame()));
  }
  StmtBldr.generateNode(D.getBindTemporaryExpr(), Pred, State);

  QualType varType = D.getBindTemporaryExpr()->getSubExpr()->getType();
  // FIXME: Currently CleanDtorState can be empty here due to temporaries being
  // bound to default parameters.
  assert(CleanDtorState.size() <= 1);
  ExplodedNode *CleanPred =
      CleanDtorState.empty() ? Pred : *CleanDtorState.begin();
  // FIXME: Inlining of temporary destructors is not supported yet anyway, so
  // we just put a NULL region for now. This will need to be changed later.
  VisitCXXDestructor(varType, nullptr, D.getBindTemporaryExpr(),
                     /*IsBase=*/false, CleanPred, Dst);
}

void ExprEngine::processCleanupTemporaryBranch(const CXXBindTemporaryExpr *BTE,
                                               NodeBuilderContext &BldCtx,
                                               ExplodedNode *Pred,
                                               ExplodedNodeSet &Dst,
                                               const CFGBlock *DstT,
                                               const CFGBlock *DstF) {
  BranchNodeBuilder TempDtorBuilder(Pred, Dst, BldCtx, DstT, DstF);
  if (Pred->getState()->contains<InitializedTemporariesSet>(
          std::make_pair(BTE, Pred->getStackFrame()))) {
    TempDtorBuilder.markInfeasible(false);
    TempDtorBuilder.generateNode(Pred->getState(), true, Pred);
  } else {
    TempDtorBuilder.markInfeasible(true);
    TempDtorBuilder.generateNode(Pred->getState(), false, Pred);
  }
}

void ExprEngine::VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *BTE,
                                           ExplodedNodeSet &PreVisit,
                                           ExplodedNodeSet &Dst) {
  if (!getAnalysisManager().options.includeTemporaryDtorsInCFG()) {
    // In case we don't have temporary destructors in the CFG, do not mark
    // the initialization - we would otherwise never clean it up.
    Dst = PreVisit;
    return;
  }
  StmtNodeBuilder StmtBldr(PreVisit, Dst, *currBldrCtx);
  for (ExplodedNode *Node : PreVisit) {
    ProgramStateRef State = Node->getState();

    if (!State->contains<InitializedTemporariesSet>(
            std::make_pair(BTE, Node->getStackFrame()))) {
      // FIXME: Currently the state might already contain the marker due to
      // incorrect handling of temporaries bound to default parameters; for
      // those, we currently skip the CXXBindTemporaryExpr but rely on adding
      // temporary destructor nodes.
      State = State->add<InitializedTemporariesSet>(
          std::make_pair(BTE, Node->getStackFrame()));
    }
    StmtBldr.generateNode(BTE, Node, State);
  }
}

void ExprEngine::Visit(const Stmt *S, ExplodedNode *Pred,
                       ExplodedNodeSet &DstTop) {
  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                S->getLocStart(),
                                "Error evaluating statement");
  ExplodedNodeSet Dst;
  StmtNodeBuilder Bldr(Pred, DstTop, *currBldrCtx);

  assert(!isa<Expr>(S) || S == cast<Expr>(S)->IgnoreParens());

  switch (S->getStmtClass()) {
    // C++ and ARC stuff we don't support yet.
    case Expr::ObjCIndirectCopyRestoreExprClass:
    case Stmt::CXXDependentScopeMemberExprClass:
    case Stmt::CXXInheritedCtorInitExprClass:
    case Stmt::CXXTryStmtClass:
    case Stmt::CXXTypeidExprClass:
    case Stmt::CXXUuidofExprClass:
    case Stmt::CXXFoldExprClass:
    case Stmt::MSPropertyRefExprClass:
    case Stmt::MSPropertySubscriptExprClass:
    case Stmt::CXXUnresolvedConstructExprClass:
    case Stmt::DependentScopeDeclRefExprClass:
    case Stmt::ArrayTypeTraitExprClass:
    case Stmt::ExpressionTraitExprClass:
    case Stmt::UnresolvedLookupExprClass:
    case Stmt::UnresolvedMemberExprClass:
    case Stmt::TypoExprClass:
    case Stmt::CXXNoexceptExprClass:
    case Stmt::PackExpansionExprClass:
    case Stmt::SubstNonTypeTemplateParmPackExprClass:
    case Stmt::FunctionParmPackExprClass:
    case Stmt::CoroutineBodyStmtClass:
    case Stmt::CoawaitExprClass:
    case Stmt::CoreturnStmtClass:
    case Stmt::CoyieldExprClass:
    case Stmt::SEHTryStmtClass:
    case Stmt::SEHExceptStmtClass:
    case Stmt::SEHLeaveStmtClass:
    case Stmt::SEHFinallyStmtClass: {
      const ExplodedNode *node = Bldr.generateSink(S, Pred, Pred->getState());
      Engine.addAbortedBlock(node, currBldrCtx->getBlock());
      break;
    }

    case Stmt::ParenExprClass:
      llvm_unreachable("ParenExprs already handled.");
    case Stmt::GenericSelectionExprClass:
      llvm_unreachable("GenericSelectionExprs already handled.");
    // Cases that should never be evaluated simply because they shouldn't
    // appear in the CFG.
    case Stmt::BreakStmtClass:
    case Stmt::CaseStmtClass:
    case Stmt::CompoundStmtClass:
    case Stmt::ContinueStmtClass:
    case Stmt::CXXForRangeStmtClass:
    case Stmt::DefaultStmtClass:
    case Stmt::DoStmtClass:
    case Stmt::ForStmtClass:
    case Stmt::GotoStmtClass:
    case Stmt::IfStmtClass:
    case Stmt::IndirectGotoStmtClass:
    case Stmt::LabelStmtClass:
    case Stmt::NoStmtClass:
    case Stmt::NullStmtClass:
    case Stmt::SwitchStmtClass:
    case Stmt::WhileStmtClass:
    case Expr::MSDependentExistsStmtClass:
    case Stmt::CapturedStmtClass:
    case Stmt::OMPParallelDirectiveClass:
    case Stmt::OMPSimdDirectiveClass:
    case Stmt::OMPForDirectiveClass:
    case Stmt::OMPForSimdDirectiveClass:
    case Stmt::OMPSectionsDirectiveClass:
    case Stmt::OMPSectionDirectiveClass:
    case Stmt::OMPSingleDirectiveClass:
    case Stmt::OMPMasterDirectiveClass:
    case Stmt::OMPCriticalDirectiveClass:
    case Stmt::OMPParallelForDirectiveClass:
    case Stmt::OMPParallelForSimdDirectiveClass:
    case Stmt::OMPParallelSectionsDirectiveClass:
    case Stmt::OMPTaskDirectiveClass:
    case Stmt::OMPTaskyieldDirectiveClass:
    case Stmt::OMPBarrierDirectiveClass:
    case Stmt::OMPTaskwaitDirectiveClass:
    case Stmt::OMPTaskgroupDirectiveClass:
    case Stmt::OMPFlushDirectiveClass:
    case Stmt::OMPOrderedDirectiveClass:
    case Stmt::OMPAtomicDirectiveClass:
    case Stmt::OMPTargetDirectiveClass:
    case Stmt::OMPTargetDataDirectiveClass:
    case Stmt::OMPTargetEnterDataDirectiveClass:
    case Stmt::OMPTargetExitDataDirectiveClass:
    case Stmt::OMPTargetParallelDirectiveClass:
    case Stmt::OMPTargetParallelForDirectiveClass:
    case Stmt::OMPTargetUpdateDirectiveClass:
    case Stmt::OMPTeamsDirectiveClass:
    case Stmt::OMPCancellationPointDirectiveClass:
    case Stmt::OMPCancelDirectiveClass:
    case Stmt::OMPTaskLoopDirectiveClass:
    case Stmt::OMPTaskLoopSimdDirectiveClass:
    case Stmt::OMPDistributeDirectiveClass:
    case Stmt::OMPDistributeParallelForDirectiveClass:
    case Stmt::OMPDistributeParallelForSimdDirectiveClass:
    case Stmt::OMPDistributeSimdDirectiveClass:
    case Stmt::OMPTargetParallelForSimdDirectiveClass:
    case Stmt::OMPTargetSimdDirectiveClass:
    case Stmt::OMPTeamsDistributeDirectiveClass:
      llvm_unreachable("Stmt should not be in analyzer evaluation loop");

    case Stmt::ObjCSubscriptRefExprClass:
    case Stmt::ObjCPropertyRefExprClass:
      llvm_unreachable("These are handled by PseudoObjectExpr");

    case Stmt::GNUNullExprClass: {
      // GNU __null is a pointer-width integer, not an actual pointer.
      ProgramStateRef state = Pred->getState();
      state = state->BindExpr(S, Pred->getLocationContext(),
                              svalBuilder.makeIntValWithPtrWidth(0, false));
      Bldr.generateNode(S, Pred, state);
      break;
    }

    case Stmt::ObjCAtSynchronizedStmtClass:
      Bldr.takeNodes(Pred);
      VisitObjCAtSynchronizedStmt(cast<ObjCAtSynchronizedStmt>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::ExprWithCleanupsClass:
      // Handled due to fully linearised CFG.
      break;

    case Stmt::CXXBindTemporaryExprClass: {
      Bldr.takeNodes(Pred);
      ExplodedNodeSet PreVisit;
      getCheckerManager().runCheckersForPreStmt(PreVisit, Pred, S, *this);
      ExplodedNodeSet Next;
      VisitCXXBindTemporaryExpr(cast<CXXBindTemporaryExpr>(S), PreVisit, Next);
      getCheckerManager().runCheckersForPostStmt(Dst, Next, S, *this);
      Bldr.addNodes(Dst);
      break;
    }

    // Cases not handled yet; but will handle some day.
    case Stmt::DesignatedInitExprClass:
    case Stmt::DesignatedInitUpdateExprClass:
    case Stmt::ExtVectorElementExprClass:
    case Stmt::ImaginaryLiteralClass:
    case Stmt::ObjCAtCatchStmtClass:
    case Stmt::ObjCAtFinallyStmtClass:
    case Stmt::ObjCAtTryStmtClass:
    case Stmt::ObjCAutoreleasePoolStmtClass:
    case Stmt::ObjCEncodeExprClass:
    case Stmt::ObjCIsaExprClass:
    case Stmt::ObjCProtocolExprClass:
    case Stmt::ObjCSelectorExprClass:
    case Stmt::ParenListExprClass:
    case Stmt::ShuffleVectorExprClass:
    case Stmt::ConvertVectorExprClass:
    case Stmt::VAArgExprClass:
    case Stmt::CUDAKernelCallExprClass:
    case Stmt::OpaqueValueExprClass:
    case Stmt::AsTypeExprClass:
      // Fall through.

    // Cases we intentionally don't evaluate, since they don't need
    // to be explicitly evaluated.
    case Stmt::PredefinedExprClass:
    case Stmt::AddrLabelExprClass:
    case Stmt::AttributedStmtClass:
    case Stmt::IntegerLiteralClass:
    case Stmt::CharacterLiteralClass:
    case Stmt::ImplicitValueInitExprClass:
    case Stmt::CXXScalarValueInitExprClass:
    case Stmt::CXXBoolLiteralExprClass:
    case Stmt::ObjCBoolLiteralExprClass:
    case Stmt::ObjCAvailabilityCheckExprClass:
    case Stmt::FloatingLiteralClass:
    case Stmt::NoInitExprClass:
    case Stmt::SizeOfPackExprClass:
    case Stmt::StringLiteralClass:
    case Stmt::ObjCStringLiteralClass:
    case Stmt::CXXPseudoDestructorExprClass:
    case Stmt::SubstNonTypeTemplateParmExprClass:
    case Stmt::CXXNullPtrLiteralExprClass:
    case Stmt::OMPArraySectionExprClass:
    case Stmt::TypeTraitExprClass: {
      Bldr.takeNodes(Pred);
      ExplodedNodeSet preVisit;
      getCheckerManager().runCheckersForPreStmt(preVisit, Pred, S, *this);
      getCheckerManager().runCheckersForPostStmt(Dst, preVisit, S, *this);
      Bldr.addNodes(Dst);
      break;
    }

    case Stmt::CXXDefaultArgExprClass:
    case Stmt::CXXDefaultInitExprClass: {
      Bldr.takeNodes(Pred);
      ExplodedNodeSet PreVisit;
      getCheckerManager().runCheckersForPreStmt(PreVisit, Pred, S, *this);

      ExplodedNodeSet Tmp;
      StmtNodeBuilder Bldr2(PreVisit, Tmp, *currBldrCtx);

      const Expr *ArgE;
      if (const CXXDefaultArgExpr *DefE = dyn_cast<CXXDefaultArgExpr>(S))
        ArgE = DefE->getExpr();
      else if (const CXXDefaultInitExpr *DefE = dyn_cast<CXXDefaultInitExpr>(S))
        ArgE = DefE->getExpr();
      else
        llvm_unreachable("unknown constant wrapper kind");

      bool IsTemporary = false;
      if (const MaterializeTemporaryExpr *MTE =
            dyn_cast<MaterializeTemporaryExpr>(ArgE)) {
        ArgE = MTE->GetTemporaryExpr();
        IsTemporary = true;
      }

      Optional<SVal> ConstantVal = svalBuilder.getConstantVal(ArgE);
      if (!ConstantVal)
        ConstantVal = UnknownVal();

      const LocationContext *LCtx = Pred->getLocationContext();
      for (ExplodedNodeSet::iterator I = PreVisit.begin(), E = PreVisit.end();
           I != E; ++I) {
        ProgramStateRef State = (*I)->getState();
        State = State->BindExpr(S, LCtx, *ConstantVal);
        if (IsTemporary)
          State = createTemporaryRegionIfNeeded(State, LCtx,
                                                cast<Expr>(S),
                                                cast<Expr>(S));
        Bldr2.generateNode(S, *I, State);
      }

      getCheckerManager().runCheckersForPostStmt(Dst, Tmp, S, *this);
      Bldr.addNodes(Dst);
      break;
    }

    // Cases we evaluate as opaque expressions, conjuring a symbol.
    case Stmt::CXXStdInitializerListExprClass:
    case Expr::ObjCArrayLiteralClass:
    case Expr::ObjCDictionaryLiteralClass:
    case Expr::ObjCBoxedExprClass: {
      Bldr.takeNodes(Pred);

      ExplodedNodeSet preVisit;
      getCheckerManager().runCheckersForPreStmt(preVisit, Pred, S, *this);

      ExplodedNodeSet Tmp;
      StmtNodeBuilder Bldr2(preVisit, Tmp, *currBldrCtx);

      const Expr *Ex = cast<Expr>(S);
      QualType resultType = Ex->getType();

      for (ExplodedNodeSet::iterator it = preVisit.begin(), et = preVisit.end();
           it != et; ++it) {
        ExplodedNode *N = *it;
        const LocationContext *LCtx = N->getLocationContext();
        SVal result = svalBuilder.conjureSymbolVal(nullptr, Ex, LCtx,
                                                   resultType,
                                                   currBldrCtx->blockCount());
        ProgramStateRef state = N->getState()->BindExpr(Ex, LCtx, result);
        Bldr2.generateNode(S, N, state);
      }

      getCheckerManager().runCheckersForPostStmt(Dst, Tmp, S, *this);
      Bldr.addNodes(Dst);
      break;
    }

    case Stmt::ArraySubscriptExprClass:
      Bldr.takeNodes(Pred);
      VisitLvalArraySubscriptExpr(cast<ArraySubscriptExpr>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::GCCAsmStmtClass:
      Bldr.takeNodes(Pred);
      VisitGCCAsmStmt(cast<GCCAsmStmt>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::MSAsmStmtClass:
      Bldr.takeNodes(Pred);
      VisitMSAsmStmt(cast<MSAsmStmt>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::BlockExprClass:
      Bldr.takeNodes(Pred);
      VisitBlockExpr(cast<BlockExpr>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::LambdaExprClass:
      if (AMgr.options.shouldInlineLambdas()) {
        Bldr.takeNodes(Pred);
        VisitLambdaExpr(cast<LambdaExpr>(S), Pred, Dst);
        Bldr.addNodes(Dst);
      } else {
        const ExplodedNode *node = Bldr.generateSink(S, Pred, Pred->getState());
        Engine.addAbortedBlock(node, currBldrCtx->getBlock());
      }
      break;

    case Stmt::BinaryOperatorClass: {
      const BinaryOperator* B = cast<BinaryOperator>(S);
      if (B->isLogicalOp()) {
        Bldr.takeNodes(Pred);
        VisitLogicalExpr(B, Pred, Dst);
        Bldr.addNodes(Dst);
        break;
      }
      else if (B->getOpcode() == BO_Comma) {
        ProgramStateRef state = Pred->getState();
        Bldr.generateNode(B, Pred,
                          state->BindExpr(B, Pred->getLocationContext(),
                                          state->getSVal(B->getRHS(),
                                                  Pred->getLocationContext())));
        break;
      }

      Bldr.takeNodes(Pred);

      if (AMgr.options.eagerlyAssumeBinOpBifurcation &&
          (B->isRelationalOp() || B->isEqualityOp())) {
        ExplodedNodeSet Tmp;
        VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Tmp);
        evalEagerlyAssumeBinOpBifurcation(Dst, Tmp, cast<Expr>(S));
      }
      else
        VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Dst);

      Bldr.addNodes(Dst);
      break;
    }

    case Stmt::CXXOperatorCallExprClass: {
      const CXXOperatorCallExpr *OCE = cast<CXXOperatorCallExpr>(S);

      // For instance method operators, make sure the 'this' argument has a
      // valid region.
      const Decl *Callee = OCE->getCalleeDecl();
      if (const CXXMethodDecl *MD = dyn_cast_or_null<CXXMethodDecl>(Callee)) {
        if (MD->isInstance()) {
          ProgramStateRef State = Pred->getState();
          const LocationContext *LCtx = Pred->getLocationContext();
          ProgramStateRef NewState =
            createTemporaryRegionIfNeeded(State, LCtx, OCE->getArg(0));
          if (NewState != State) {
            Pred = Bldr.generateNode(OCE, Pred, NewState, /*Tag=*/nullptr,
                                     ProgramPoint::PreStmtKind);
            // Did we cache out?
            if (!Pred)
              break;
          }
        }
      }
      // FALLTHROUGH
    }
    case Stmt::CallExprClass:
    case Stmt::CXXMemberCallExprClass:
    case Stmt::UserDefinedLiteralClass: {
      Bldr.takeNodes(Pred);
      VisitCallExpr(cast<CallExpr>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;
    }

    case Stmt::CXXCatchStmtClass: {
      Bldr.takeNodes(Pred);
      VisitCXXCatchStmt(cast<CXXCatchStmt>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;
    }

    case Stmt::CXXTemporaryObjectExprClass:
    case Stmt::CXXConstructExprClass: {
      Bldr.takeNodes(Pred);
      VisitCXXConstructExpr(cast<CXXConstructExpr>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;
    }

    case Stmt::CXXNewExprClass: {
      Bldr.takeNodes(Pred);
      ExplodedNodeSet PostVisit;
      VisitCXXNewExpr(cast<CXXNewExpr>(S), Pred, PostVisit);
      getCheckerManager().runCheckersForPostStmt(Dst, PostVisit, S, *this);
      Bldr.addNodes(Dst);
      break;
    }

    case Stmt::CXXDeleteExprClass: {
      Bldr.takeNodes(Pred);
      ExplodedNodeSet PreVisit;
      const CXXDeleteExpr *CDE = cast<CXXDeleteExpr>(S);
      getCheckerManager().runCheckersForPreStmt(PreVisit, Pred, S, *this);

      for (ExplodedNodeSet::iterator i = PreVisit.begin(),
                                     e = PreVisit.end(); i != e ; ++i)
        VisitCXXDeleteExpr(CDE, *i, Dst);

      Bldr.addNodes(Dst);
      break;
    }
      // FIXME: ChooseExpr is really a constant.  We need to fix
      //        the CFG do not model them as explicit control-flow.

    case Stmt::ChooseExprClass: { // __builtin_choose_expr
      Bldr.takeNodes(Pred);
      const ChooseExpr *C = cast<ChooseExpr>(S);
      VisitGuardedExpr(C, C->getLHS(), C->getRHS(), Pred, Dst);
      Bldr.addNodes(Dst);
      break;
    }

    case Stmt::CompoundAssignOperatorClass:
      Bldr.takeNodes(Pred);
      VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::CompoundLiteralExprClass:
      Bldr.takeNodes(Pred);
      VisitCompoundLiteralExpr(cast<CompoundLiteralExpr>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::BinaryConditionalOperatorClass:
    case Stmt::ConditionalOperatorClass: { // '?' operator
      Bldr.takeNodes(Pred);
      const AbstractConditionalOperator *C
        = cast<AbstractConditionalOperator>(S);
      VisitGuardedExpr(C, C->getTrueExpr(), C->getFalseExpr(), Pred, Dst);
      Bldr.addNodes(Dst);
      break;
    }

    case Stmt::CXXThisExprClass:
      Bldr.takeNodes(Pred);
      VisitCXXThisExpr(cast<CXXThisExpr>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::DeclRefExprClass: {
      Bldr.takeNodes(Pred);
      const DeclRefExpr *DE = cast<DeclRefExpr>(S);
      VisitCommonDeclRefExpr(DE, DE->getDecl(), Pred, Dst);
      Bldr.addNodes(Dst);
      break;
    }

    case Stmt::DeclStmtClass:
      Bldr.takeNodes(Pred);
      VisitDeclStmt(cast<DeclStmt>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::ImplicitCastExprClass:
    case Stmt::CStyleCastExprClass:
    case Stmt::CXXStaticCastExprClass:
    case Stmt::CXXDynamicCastExprClass:
    case Stmt::CXXReinterpretCastExprClass:
    case Stmt::CXXConstCastExprClass:
    case Stmt::CXXFunctionalCastExprClass:
    case Stmt::ObjCBridgedCastExprClass: {
      Bldr.takeNodes(Pred);
      const CastExpr *C = cast<CastExpr>(S);
      ExplodedNodeSet dstExpr;
      VisitCast(C, C->getSubExpr(), Pred, dstExpr);

      // Handle the postvisit checks.
      getCheckerManager().runCheckersForPostStmt(Dst, dstExpr, C, *this);
      Bldr.addNodes(Dst);
      break;
    }

    case Expr::MaterializeTemporaryExprClass: {
      Bldr.takeNodes(Pred);
      const MaterializeTemporaryExpr *MTE = cast<MaterializeTemporaryExpr>(S);
      CreateCXXTemporaryObject(MTE, Pred, Dst);
      Bldr.addNodes(Dst);
      break;
    }

    case Stmt::InitListExprClass:
      Bldr.takeNodes(Pred);
      VisitInitListExpr(cast<InitListExpr>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::MemberExprClass:
      Bldr.takeNodes(Pred);
      VisitMemberExpr(cast<MemberExpr>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::AtomicExprClass:
      Bldr.takeNodes(Pred);
      VisitAtomicExpr(cast<AtomicExpr>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::ObjCIvarRefExprClass:
      Bldr.takeNodes(Pred);
      VisitLvalObjCIvarRefExpr(cast<ObjCIvarRefExpr>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::ObjCForCollectionStmtClass:
      Bldr.takeNodes(Pred);
      VisitObjCForCollectionStmt(cast<ObjCForCollectionStmt>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::ObjCMessageExprClass:
      Bldr.takeNodes(Pred);
      VisitObjCMessage(cast<ObjCMessageExpr>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::ObjCAtThrowStmtClass:
    case Stmt::CXXThrowExprClass:
      // FIXME: This is not complete.  We basically treat @throw as
      // an abort.
      Bldr.generateSink(S, Pred, Pred->getState());
      break;

    case Stmt::ReturnStmtClass:
      Bldr.takeNodes(Pred);
      VisitReturnStmt(cast<ReturnStmt>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::OffsetOfExprClass:
      Bldr.takeNodes(Pred);
      VisitOffsetOfExpr(cast<OffsetOfExpr>(S), Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::UnaryExprOrTypeTraitExprClass:
      Bldr.takeNodes(Pred);
      VisitUnaryExprOrTypeTraitExpr(cast<UnaryExprOrTypeTraitExpr>(S),
                                    Pred, Dst);
      Bldr.addNodes(Dst);
      break;

    case Stmt::StmtExprClass: {
      const StmtExpr *SE = cast<StmtExpr>(S);

      if (SE->getSubStmt()->body_empty()) {
        // Empty statement expression.
        assert(SE->getType() == getContext().VoidTy
               && "Empty statement expression must have void type.");
        break;
      }

      if (Expr *LastExpr = dyn_cast<Expr>(*SE->getSubStmt()->body_rbegin())) {
        ProgramStateRef state = Pred->getState();
        Bldr.generateNode(SE, Pred,
                          state->BindExpr(SE, Pred->getLocationContext(),
                                          state->getSVal(LastExpr,
                                                  Pred->getLocationContext())));
      }
      break;
    }

    case Stmt::UnaryOperatorClass: {
      Bldr.takeNodes(Pred);
      const UnaryOperator *U = cast<UnaryOperator>(S);
      if (AMgr.options.eagerlyAssumeBinOpBifurcation && (U->getOpcode() == UO_LNot)) {
        ExplodedNodeSet Tmp;
        VisitUnaryOperator(U, Pred, Tmp);
        evalEagerlyAssumeBinOpBifurcation(Dst, Tmp, U);
      }
      else
        VisitUnaryOperator(U, Pred, Dst);
      Bldr.addNodes(Dst);
      break;
    }

    case Stmt::PseudoObjectExprClass: {
      Bldr.takeNodes(Pred);
      ProgramStateRef state = Pred->getState();
      const PseudoObjectExpr *PE = cast<PseudoObjectExpr>(S);
      if (const Expr *Result = PE->getResultExpr()) {
        SVal V = state->getSVal(Result, Pred->getLocationContext());
        Bldr.generateNode(S, Pred,
                          state->BindExpr(S, Pred->getLocationContext(), V));
      }
      else
        Bldr.generateNode(S, Pred,
                          state->BindExpr(S, Pred->getLocationContext(),
                                                   UnknownVal()));

      Bldr.addNodes(Dst);
      break;
    }
  }
}

bool ExprEngine::replayWithoutInlining(ExplodedNode *N,
                                       const LocationContext *CalleeLC) {
  const StackFrameContext *CalleeSF = CalleeLC->getCurrentStackFrame();
  const StackFrameContext *CallerSF = CalleeSF->getParent()->getCurrentStackFrame();
  assert(CalleeSF && CallerSF);
  ExplodedNode *BeforeProcessingCall = nullptr;
  const Stmt *CE = CalleeSF->getCallSite();

  // Find the first node before we started processing the call expression.
  while (N) {
    ProgramPoint L = N->getLocation();
    BeforeProcessingCall = N;
    N = N->pred_empty() ? nullptr : *(N->pred_begin());

    // Skip the nodes corresponding to the inlined code.
    if (L.getLocationContext()->getCurrentStackFrame() != CallerSF)
      continue;
    // We reached the caller. Find the node right before we started
    // processing the call.
    if (L.isPurgeKind())
      continue;
    if (L.getAs<PreImplicitCall>())
      continue;
    if (L.getAs<CallEnter>())
      continue;
    if (Optional<StmtPoint> SP = L.getAs<StmtPoint>())
      if (SP->getStmt() == CE)
        continue;
    break;
  }

  if (!BeforeProcessingCall)
    return false;

  // TODO: Clean up the unneeded nodes.

  // Build an Epsilon node from which we will restart the analyzes.
  // Note that CE is permitted to be NULL!
  ProgramPoint NewNodeLoc =
               EpsilonPoint(BeforeProcessingCall->getLocationContext(), CE);
  // Add the special flag to GDM to signal retrying with no inlining.
  // Note, changing the state ensures that we are not going to cache out.
  ProgramStateRef NewNodeState = BeforeProcessingCall->getState();
  NewNodeState =
    NewNodeState->set<ReplayWithoutInlining>(const_cast<Stmt *>(CE));

  // Make the new node a successor of BeforeProcessingCall.
  bool IsNew = false;
  ExplodedNode *NewNode = G.getNode(NewNodeLoc, NewNodeState, false, &IsNew);
  // We cached out at this point. Caching out is common due to us backtracking
  // from the inlined function, which might spawn several paths.
  if (!IsNew)
    return true;

  NewNode->addPredecessor(BeforeProcessingCall, G);

  // Add the new node to the work list.
  Engine.enqueueStmtNode(NewNode, CalleeSF->getCallSiteBlock(),
                                  CalleeSF->getIndex());
  NumTimesRetriedWithoutInlining++;
  return true;
}

/// Block entrance.  (Update counters).
void ExprEngine::processCFGBlockEntrance(const BlockEdge &L,
                                         NodeBuilderWithSinks &nodeBuilder,
                                         ExplodedNode *Pred) {
  PrettyStackTraceLocationContext CrashInfo(Pred->getLocationContext());

  // If this block is terminated by a loop and it has already been visited the
  // maximum number of times, widen the loop.
  unsigned int BlockCount = nodeBuilder.getContext().blockCount();
  if (BlockCount == AMgr.options.maxBlockVisitOnPath - 1 &&
      AMgr.options.shouldWidenLoops()) {
    const Stmt *Term = nodeBuilder.getContext().getBlock()->getTerminator();
    if (!(Term &&
          (isa<ForStmt>(Term) || isa<WhileStmt>(Term) || isa<DoStmt>(Term))))
      return;
    // Widen.
    const LocationContext *LCtx = Pred->getLocationContext();
    ProgramStateRef WidenedState =
        getWidenedLoopState(Pred->getState(), LCtx, BlockCount, Term);
    nodeBuilder.generateNode(WidenedState, Pred);
    return;
  }

  // FIXME: Refactor this into a checker.
  if (BlockCount >= AMgr.options.maxBlockVisitOnPath) {
    static SimpleProgramPointTag tag(TagProviderName, "Block count exceeded");
    const ExplodedNode *Sink =
                   nodeBuilder.generateSink(Pred->getState(), Pred, &tag);

    // Check if we stopped at the top level function or not.
    // Root node should have the location context of the top most function.
    const LocationContext *CalleeLC = Pred->getLocation().getLocationContext();
    const LocationContext *CalleeSF = CalleeLC->getCurrentStackFrame();
    const LocationContext *RootLC =
                        (*G.roots_begin())->getLocation().getLocationContext();
    if (RootLC->getCurrentStackFrame() != CalleeSF) {
      Engine.FunctionSummaries->markReachedMaxBlockCount(CalleeSF->getDecl());

      // Re-run the call evaluation without inlining it, by storing the
      // no-inlining policy in the state and enqueuing the new work item on
      // the list. Replay should almost never fail. Use the stats to catch it
      // if it does.
      if ((!AMgr.options.NoRetryExhausted &&
           replayWithoutInlining(Pred, CalleeLC)))
        return;
      NumMaxBlockCountReachedInInlined++;
    } else
      NumMaxBlockCountReached++;

    // Make sink nodes as exhausted(for stats) only if retry failed.
    Engine.blocksExhausted.push_back(std::make_pair(L, Sink));
  }
}

//===----------------------------------------------------------------------===//
// Branch processing.
//===----------------------------------------------------------------------===//

/// RecoverCastedSymbol - A helper function for ProcessBranch that is used
/// to try to recover some path-sensitivity for casts of symbolic
/// integers that promote their values (which are currently not tracked well).
/// This function returns the SVal bound to Condition->IgnoreCasts if all the
//  cast(s) did was sign-extend the original value.
static SVal RecoverCastedSymbol(ProgramStateManager& StateMgr,
                                ProgramStateRef state,
                                const Stmt *Condition,
                                const LocationContext *LCtx,
                                ASTContext &Ctx) {

  const Expr *Ex = dyn_cast<Expr>(Condition);
  if (!Ex)
    return UnknownVal();

  uint64_t bits = 0;
  bool bitsInit = false;

  while (const CastExpr *CE = dyn_cast<CastExpr>(Ex)) {
    QualType T = CE->getType();

    if (!T->isIntegralOrEnumerationType())
      return UnknownVal();

    uint64_t newBits = Ctx.getTypeSize(T);
    if (!bitsInit || newBits < bits) {
      bitsInit = true;
      bits = newBits;
    }

    Ex = CE->getSubExpr();
  }

  // We reached a non-cast.  Is it a symbolic value?
  QualType T = Ex->getType();

  if (!bitsInit || !T->isIntegralOrEnumerationType() ||
      Ctx.getTypeSize(T) > bits)
    return UnknownVal();

  return state->getSVal(Ex, LCtx);
}

#ifndef NDEBUG
static const Stmt *getRightmostLeaf(const Stmt *Condition) {
  while (Condition) {
    const BinaryOperator *BO = dyn_cast<BinaryOperator>(Condition);
    if (!BO || !BO->isLogicalOp()) {
      return Condition;
    }
    Condition = BO->getRHS()->IgnoreParens();
  }
  return nullptr;
}
#endif

// Returns the condition the branch at the end of 'B' depends on and whose value
// has been evaluated within 'B'.
// In most cases, the terminator condition of 'B' will be evaluated fully in
// the last statement of 'B'; in those cases, the resolved condition is the
// given 'Condition'.
// If the condition of the branch is a logical binary operator tree, the CFG is
// optimized: in that case, we know that the expression formed by all but the
// rightmost leaf of the logical binary operator tree must be true, and thus
// the branch condition is at this point equivalent to the truth value of that
// rightmost leaf; the CFG block thus only evaluates this rightmost leaf
// expression in its final statement. As the full condition in that case was
// not evaluated, and is thus not in the SVal cache, we need to use that leaf
// expression to evaluate the truth value of the condition in the current state
// space.
static const Stmt *ResolveCondition(const Stmt *Condition,
                                    const CFGBlock *B) {
  if (const Expr *Ex = dyn_cast<Expr>(Condition))
    Condition = Ex->IgnoreParens();

  const BinaryOperator *BO = dyn_cast<BinaryOperator>(Condition);
  if (!BO || !BO->isLogicalOp())
    return Condition;

  assert(!B->getTerminator().isTemporaryDtorsBranch() &&
         "Temporary destructor branches handled by processBindTemporary.");

  // For logical operations, we still have the case where some branches
  // use the traditional "merge" approach and others sink the branch
  // directly into the basic blocks representing the logical operation.
  // We need to distinguish between those two cases here.

  // The invariants are still shifting, but it is possible that the
  // last element in a CFGBlock is not a CFGStmt.  Look for the last
  // CFGStmt as the value of the condition.
  CFGBlock::const_reverse_iterator I = B->rbegin(), E = B->rend();
  for (; I != E; ++I) {
    CFGElement Elem = *I;
    Optional<CFGStmt> CS = Elem.getAs<CFGStmt>();
    if (!CS)
      continue;
    const Stmt *LastStmt = CS->getStmt();
    assert(LastStmt == Condition || LastStmt == getRightmostLeaf(Condition));
    return LastStmt;
  }
  llvm_unreachable("could not resolve condition");
}

void ExprEngine::processBranch(const Stmt *Condition, const Stmt *Term,
                               NodeBuilderContext& BldCtx,
                               ExplodedNode *Pred,
                               ExplodedNodeSet &Dst,
                               const CFGBlock *DstT,
                               const CFGBlock *DstF) {
  assert((!Condition || !isa<CXXBindTemporaryExpr>(Condition)) &&
         "CXXBindTemporaryExprs are handled by processBindTemporary.");
  const LocationContext *LCtx = Pred->getLocationContext();
  PrettyStackTraceLocationContext StackCrashInfo(LCtx);
  currBldrCtx = &BldCtx;

  // Check for NULL conditions; e.g. "for(;;)"
  if (!Condition) {
    BranchNodeBuilder NullCondBldr(Pred, Dst, BldCtx, DstT, DstF);
    NullCondBldr.markInfeasible(false);
    NullCondBldr.generateNode(Pred->getState(), true, Pred);
    return;
  }

  if (const Expr *Ex = dyn_cast<Expr>(Condition))
    Condition = Ex->IgnoreParens();

  Condition = ResolveCondition(Condition, BldCtx.getBlock());
  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                Condition->getLocStart(),
                                "Error evaluating branch");

  ExplodedNodeSet CheckersOutSet;
  getCheckerManager().runCheckersForBranchCondition(Condition, CheckersOutSet,
                                                    Pred, *this);
  // We generated only sinks.
  if (CheckersOutSet.empty())
    return;

  BranchNodeBuilder builder(CheckersOutSet, Dst, BldCtx, DstT, DstF);
  for (NodeBuilder::iterator I = CheckersOutSet.begin(),
                             E = CheckersOutSet.end(); E != I; ++I) {
    ExplodedNode *PredI = *I;

    if (PredI->isSink())
      continue;

    ProgramStateRef PrevState = PredI->getState();
    SVal X = PrevState->getSVal(Condition, PredI->getLocationContext());

    if (X.isUnknownOrUndef()) {
      // Give it a chance to recover from unknown.
      if (const Expr *Ex = dyn_cast<Expr>(Condition)) {
        if (Ex->getType()->isIntegralOrEnumerationType()) {
          // Try to recover some path-sensitivity.  Right now casts of symbolic
          // integers that promote their values are currently not tracked well.
          // If 'Condition' is such an expression, try and recover the
          // underlying value and use that instead.
          SVal recovered = RecoverCastedSymbol(getStateManager(),
                                               PrevState, Condition,
                                               PredI->getLocationContext(),
                                               getContext());

          if (!recovered.isUnknown()) {
            X = recovered;
          }
        }
      }
    }

    // If the condition is still unknown, give up.
    if (X.isUnknownOrUndef()) {
      builder.generateNode(PrevState, true, PredI);
      builder.generateNode(PrevState, false, PredI);
      continue;
    }

    DefinedSVal V = X.castAs<DefinedSVal>();

    ProgramStateRef StTrue, StFalse;
    std::tie(StTrue, StFalse) = PrevState->assume(V);

    // Process the true branch.
    if (builder.isFeasible(true)) {
      if (StTrue)
        builder.generateNode(StTrue, true, PredI);
      else
        builder.markInfeasible(true);
    }

    // Process the false branch.
    if (builder.isFeasible(false)) {
      if (StFalse)
        builder.generateNode(StFalse, false, PredI);
      else
        builder.markInfeasible(false);
    }
  }
  currBldrCtx = nullptr;
}

/// The GDM component containing the set of global variables which have been
/// previously initialized with explicit initializers.
REGISTER_TRAIT_WITH_PROGRAMSTATE(InitializedGlobalsSet,
                                 llvm::ImmutableSet<const VarDecl *>)

void ExprEngine::processStaticInitializer(const DeclStmt *DS,
                                          NodeBuilderContext &BuilderCtx,
                                          ExplodedNode *Pred,
                                          clang::ento::ExplodedNodeSet &Dst,
                                          const CFGBlock *DstT,
                                          const CFGBlock *DstF) {
  PrettyStackTraceLocationContext CrashInfo(Pred->getLocationContext());
  currBldrCtx = &BuilderCtx;

  const VarDecl *VD = cast<VarDecl>(DS->getSingleDecl());
  ProgramStateRef state = Pred->getState();
  bool initHasRun = state->contains<InitializedGlobalsSet>(VD);
  BranchNodeBuilder builder(Pred, Dst, BuilderCtx, DstT, DstF);

  if (!initHasRun) {
    state = state->add<InitializedGlobalsSet>(VD);
  }

  builder.generateNode(state, initHasRun, Pred);
  builder.markInfeasible(!initHasRun);

  currBldrCtx = nullptr;
}

/// processIndirectGoto - Called by CoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a computed goto jump.
void ExprEngine::processIndirectGoto(IndirectGotoNodeBuilder &builder) {

  ProgramStateRef state = builder.getState();
  SVal V = state->getSVal(builder.getTarget(), builder.getLocationContext());

  // Three possibilities:
  //
  //   (1) We know the computed label.
  //   (2) The label is NULL (or some other constant), or Undefined.
  //   (3) We have no clue about the label.  Dispatch to all targets.
  //

  typedef IndirectGotoNodeBuilder::iterator iterator;

  if (Optional<loc::GotoLabel> LV = V.getAs<loc::GotoLabel>()) {
    const LabelDecl *L = LV->getLabel();

    for (iterator I = builder.begin(), E = builder.end(); I != E; ++I) {
      if (I.getLabel() == L) {
        builder.generateNode(I, state);
        return;
      }
    }

    llvm_unreachable("No block with label.");
  }

  if (V.getAs<loc::ConcreteInt>() || V.getAs<UndefinedVal>()) {
    // Dispatch to the first target and mark it as a sink.
    //ExplodedNode* N = builder.generateNode(builder.begin(), state, true);
    // FIXME: add checker visit.
    //    UndefBranches.insert(N);
    return;
  }

  // This is really a catch-all.  We don't support symbolics yet.
  // FIXME: Implement dispatch for symbolic pointers.

  for (iterator I=builder.begin(), E=builder.end(); I != E; ++I)
    builder.generateNode(I, state);
}

#if 0
static bool stackFrameDoesNotContainInitializedTemporaries(ExplodedNode &Pred) {
  const StackFrameContext* Frame = Pred.getStackFrame();
  const llvm::ImmutableSet<CXXBindTemporaryContext> &Set =
      Pred.getState()->get<InitializedTemporariesSet>();
  return std::find_if(Set.begin(), Set.end(),
                      [&](const CXXBindTemporaryContext &Ctx) {
                        if (Ctx.second == Frame) {
                          Ctx.first->dump();
                          llvm::errs() << "\n";
                        }
           return Ctx.second == Frame;
         }) == Set.end();
}
#endif

void ExprEngine::processBeginOfFunction(NodeBuilderContext &BC,
                                        ExplodedNode *Pred,
                                        ExplodedNodeSet &Dst,
                                        const BlockEdge &L) {
  SaveAndRestore<const NodeBuilderContext *> NodeContextRAII(currBldrCtx, &BC);
  getCheckerManager().runCheckersForBeginFunction(Dst, L, Pred, *this);
}

/// ProcessEndPath - Called by CoreEngine.  Used to generate end-of-path
///  nodes when the control reaches the end of a function.
void ExprEngine::processEndOfFunction(NodeBuilderContext& BC,
                                      ExplodedNode *Pred) {
  // FIXME: Assert that stackFrameDoesNotContainInitializedTemporaries(*Pred)).
  // We currently cannot enable this assert, as lifetime extended temporaries
  // are not modelled correctly.
  PrettyStackTraceLocationContext CrashInfo(Pred->getLocationContext());
  StateMgr.EndPath(Pred->getState());

  ExplodedNodeSet Dst;
  if (Pred->getLocationContext()->inTopFrame()) {
    // Remove dead symbols.
    ExplodedNodeSet AfterRemovedDead;
    removeDeadOnEndOfFunction(BC, Pred, AfterRemovedDead);

    // Notify checkers.
    for (ExplodedNodeSet::iterator I = AfterRemovedDead.begin(),
        E = AfterRemovedDead.end(); I != E; ++I) {
      getCheckerManager().runCheckersForEndFunction(BC, Dst, *I, *this);
    }
  } else {
    getCheckerManager().runCheckersForEndFunction(BC, Dst, Pred, *this);
  }

  Engine.enqueueEndOfFunction(Dst);
}

/// ProcessSwitch - Called by CoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a switch statement.
void ExprEngine::processSwitch(SwitchNodeBuilder& builder) {
  typedef SwitchNodeBuilder::iterator iterator;
  ProgramStateRef state = builder.getState();
  const Expr *CondE = builder.getCondition();
  SVal  CondV_untested = state->getSVal(CondE, builder.getLocationContext());

  if (CondV_untested.isUndef()) {
    //ExplodedNode* N = builder.generateDefaultCaseNode(state, true);
    // FIXME: add checker
    //UndefBranches.insert(N);

    return;
  }
  DefinedOrUnknownSVal CondV = CondV_untested.castAs<DefinedOrUnknownSVal>();

  ProgramStateRef DefaultSt = state;

  iterator I = builder.begin(), EI = builder.end();
  bool defaultIsFeasible = I == EI;

  for ( ; I != EI; ++I) {
    // Successor may be pruned out during CFG construction.
    if (!I.getBlock())
      continue;

    const CaseStmt *Case = I.getCase();

    // Evaluate the LHS of the case value.
    llvm::APSInt V1 = Case->getLHS()->EvaluateKnownConstInt(getContext());
    assert(V1.getBitWidth() == getContext().getTypeSize(CondE->getType()));

    // Get the RHS of the case, if it exists.
    llvm::APSInt V2;
    if (const Expr *E = Case->getRHS())
      V2 = E->EvaluateKnownConstInt(getContext());
    else
      V2 = V1;

    ProgramStateRef StateCase;
    if (Optional<NonLoc> NL = CondV.getAs<NonLoc>())
      std::tie(StateCase, DefaultSt) =
          DefaultSt->assumeWithinInclusiveRange(*NL, V1, V2);
    else // UnknownVal
      StateCase = DefaultSt;

    if (StateCase)
      builder.generateCaseStmtNode(I, StateCase);

    // Now "assume" that the case doesn't match.  Add this state
    // to the default state (if it is feasible).
    if (DefaultSt)
      defaultIsFeasible = true;
    else {
      defaultIsFeasible = false;
      break;
    }
  }

  if (!defaultIsFeasible)
    return;

  // If we have switch(enum value), the default branch is not
  // feasible if all of the enum constants not covered by 'case:' statements
  // are not feasible values for the switch condition.
  //
  // Note that this isn't as accurate as it could be.  Even if there isn't
  // a case for a particular enum value as long as that enum value isn't
  // feasible then it shouldn't be considered for making 'default:' reachable.
  const SwitchStmt *SS = builder.getSwitch();
  const Expr *CondExpr = SS->getCond()->IgnoreParenImpCasts();
  if (CondExpr->getType()->getAs<EnumType>()) {
    if (SS->isAllEnumCasesCovered())
      return;
  }

  builder.generateDefaultCaseNode(DefaultSt);
}

//===----------------------------------------------------------------------===//
// Transfer functions: Loads and stores.
//===----------------------------------------------------------------------===//

void ExprEngine::VisitCommonDeclRefExpr(const Expr *Ex, const NamedDecl *D,
                                        ExplodedNode *Pred,
                                        ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);

  ProgramStateRef state = Pred->getState();
  const LocationContext *LCtx = Pred->getLocationContext();

  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    // C permits "extern void v", and if you cast the address to a valid type,
    // you can even do things with it. We simply pretend
    assert(Ex->isGLValue() || VD->getType()->isVoidType());
    const LocationContext *LocCtxt = Pred->getLocationContext();
    const Decl *D = LocCtxt->getDecl();
    const auto *MD = D ? dyn_cast<CXXMethodDecl>(D) : nullptr;
    const auto *DeclRefEx = dyn_cast<DeclRefExpr>(Ex);
    SVal V;
    bool IsReference;
    if (AMgr.options.shouldInlineLambdas() && DeclRefEx &&
        DeclRefEx->refersToEnclosingVariableOrCapture() && MD &&
        MD->getParent()->isLambda()) {
      // Lookup the field of the lambda.
      const CXXRecordDecl *CXXRec = MD->getParent();
      llvm::DenseMap<const VarDecl *, FieldDecl *> LambdaCaptureFields;
      FieldDecl *LambdaThisCaptureField;
      CXXRec->getCaptureFields(LambdaCaptureFields, LambdaThisCaptureField);
      const FieldDecl *FD = LambdaCaptureFields[VD];
      if (!FD) {
        // When a constant is captured, sometimes no corresponding field is
        // created in the lambda object.
        assert(VD->getType().isConstQualified());
        V = state->getLValue(VD, LocCtxt);
        IsReference = false;
      } else {
        Loc CXXThis =
            svalBuilder.getCXXThis(MD, LocCtxt->getCurrentStackFrame());
        SVal CXXThisVal = state->getSVal(CXXThis);
        V = state->getLValue(FD, CXXThisVal);
        IsReference = FD->getType()->isReferenceType();
      }
    } else {
      V = state->getLValue(VD, LocCtxt);
      IsReference = VD->getType()->isReferenceType();
    }

    // For references, the 'lvalue' is the pointer address stored in the
    // reference region.
    if (IsReference) {
      if (const MemRegion *R = V.getAsRegion())
        V = state->getSVal(R);
      else
        V = UnknownVal();
    }

    Bldr.generateNode(Ex, Pred, state->BindExpr(Ex, LCtx, V), nullptr,
                      ProgramPoint::PostLValueKind);
    return;
  }
  if (const EnumConstantDecl *ED = dyn_cast<EnumConstantDecl>(D)) {
    assert(!Ex->isGLValue());
    SVal V = svalBuilder.makeIntVal(ED->getInitVal());
    Bldr.generateNode(Ex, Pred, state->BindExpr(Ex, LCtx, V));
    return;
  }
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    SVal V = svalBuilder.getFunctionPointer(FD);
    Bldr.generateNode(Ex, Pred, state->BindExpr(Ex, LCtx, V), nullptr,
                      ProgramPoint::PostLValueKind);
    return;
  }
  if (isa<FieldDecl>(D)) {
    // FIXME: Compute lvalue of field pointers-to-member.
    // Right now we just use a non-null void pointer, so that it gives proper
    // results in boolean contexts.
    SVal V = svalBuilder.conjureSymbolVal(Ex, LCtx, getContext().VoidPtrTy,
                                          currBldrCtx->blockCount());
    state = state->assume(V.castAs<DefinedOrUnknownSVal>(), true);
    Bldr.generateNode(Ex, Pred, state->BindExpr(Ex, LCtx, V), nullptr,
		      ProgramPoint::PostLValueKind);
    return;
  }

  llvm_unreachable("Support for this Decl not implemented.");
}

/// VisitArraySubscriptExpr - Transfer function for array accesses
void ExprEngine::VisitLvalArraySubscriptExpr(const ArraySubscriptExpr *A,
                                             ExplodedNode *Pred,
                                             ExplodedNodeSet &Dst){

  const Expr *Base = A->getBase()->IgnoreParens();
  const Expr *Idx  = A->getIdx()->IgnoreParens();

  ExplodedNodeSet checkerPreStmt;
  getCheckerManager().runCheckersForPreStmt(checkerPreStmt, Pred, A, *this);

  StmtNodeBuilder Bldr(checkerPreStmt, Dst, *currBldrCtx);
  assert(A->isGLValue() ||
          (!AMgr.getLangOpts().CPlusPlus &&
           A->getType().isCForbiddenLValueType()));

  for (ExplodedNodeSet::iterator it = checkerPreStmt.begin(),
                                 ei = checkerPreStmt.end(); it != ei; ++it) {
    const LocationContext *LCtx = (*it)->getLocationContext();
    ProgramStateRef state = (*it)->getState();
    SVal V = state->getLValue(A->getType(),
                              state->getSVal(Idx, LCtx),
                              state->getSVal(Base, LCtx));
    Bldr.generateNode(A, *it, state->BindExpr(A, LCtx, V), nullptr,
                      ProgramPoint::PostLValueKind);
  }
}

/// VisitMemberExpr - Transfer function for member expressions.
void ExprEngine::VisitMemberExpr(const MemberExpr *M, ExplodedNode *Pred,
                                 ExplodedNodeSet &Dst) {

  // FIXME: Prechecks eventually go in ::Visit().
  ExplodedNodeSet CheckedSet;
  getCheckerManager().runCheckersForPreStmt(CheckedSet, Pred, M, *this);

  ExplodedNodeSet EvalSet;
  ValueDecl *Member = M->getMemberDecl();

  // Handle static member variables and enum constants accessed via
  // member syntax.
  if (isa<VarDecl>(Member) || isa<EnumConstantDecl>(Member)) {
    ExplodedNodeSet Dst;
    for (ExplodedNodeSet::iterator I = CheckedSet.begin(), E = CheckedSet.end();
         I != E; ++I) {
      VisitCommonDeclRefExpr(M, Member, Pred, EvalSet);
    }
  } else {
    StmtNodeBuilder Bldr(CheckedSet, EvalSet, *currBldrCtx);
    ExplodedNodeSet Tmp;

    for (ExplodedNodeSet::iterator I = CheckedSet.begin(), E = CheckedSet.end();
         I != E; ++I) {
      ProgramStateRef state = (*I)->getState();
      const LocationContext *LCtx = (*I)->getLocationContext();
      Expr *BaseExpr = M->getBase();

      // Handle C++ method calls.
      if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(Member)) {
        if (MD->isInstance())
          state = createTemporaryRegionIfNeeded(state, LCtx, BaseExpr);

        SVal MDVal = svalBuilder.getFunctionPointer(MD);
        state = state->BindExpr(M, LCtx, MDVal);

        Bldr.generateNode(M, *I, state);
        continue;
      }

      // Handle regular struct fields / member variables.
      state = createTemporaryRegionIfNeeded(state, LCtx, BaseExpr);
      SVal baseExprVal = state->getSVal(BaseExpr, LCtx);

      FieldDecl *field = cast<FieldDecl>(Member);
      SVal L = state->getLValue(field, baseExprVal);

      if (M->isGLValue() || M->getType()->isArrayType()) {
        // We special-case rvalues of array type because the analyzer cannot
        // reason about them, since we expect all regions to be wrapped in Locs.
        // We instead treat these as lvalues and assume that they will decay to
        // pointers as soon as they are used.
        if (!M->isGLValue()) {
          assert(M->getType()->isArrayType());
          const ImplicitCastExpr *PE =
            dyn_cast<ImplicitCastExpr>((*I)->getParentMap().getParentIgnoreParens(M));
          if (!PE || PE->getCastKind() != CK_ArrayToPointerDecay) {
            llvm_unreachable("should always be wrapped in ArrayToPointerDecay");
          }
        }

        if (field->getType()->isReferenceType()) {
          if (const MemRegion *R = L.getAsRegion())
            L = state->getSVal(R);
          else
            L = UnknownVal();
        }

        Bldr.generateNode(M, *I, state->BindExpr(M, LCtx, L), nullptr,
                          ProgramPoint::PostLValueKind);
      } else {
        Bldr.takeNodes(*I);
        evalLoad(Tmp, M, M, *I, state, L);
        Bldr.addNodes(Tmp);
      }
    }
  }

  getCheckerManager().runCheckersForPostStmt(Dst, EvalSet, M, *this);
}

void ExprEngine::VisitAtomicExpr(const AtomicExpr *AE, ExplodedNode *Pred,
                                 ExplodedNodeSet &Dst) {
  ExplodedNodeSet AfterPreSet;
  getCheckerManager().runCheckersForPreStmt(AfterPreSet, Pred, AE, *this);

  // For now, treat all the arguments to C11 atomics as escaping.
  // FIXME: Ideally we should model the behavior of the atomics precisely here.

  ExplodedNodeSet AfterInvalidateSet;
  StmtNodeBuilder Bldr(AfterPreSet, AfterInvalidateSet, *currBldrCtx);

  for (ExplodedNodeSet::iterator I = AfterPreSet.begin(), E = AfterPreSet.end();
       I != E; ++I) {
    ProgramStateRef State = (*I)->getState();
    const LocationContext *LCtx = (*I)->getLocationContext();

    SmallVector<SVal, 8> ValuesToInvalidate;
    for (unsigned SI = 0, Count = AE->getNumSubExprs(); SI != Count; SI++) {
      const Expr *SubExpr = AE->getSubExprs()[SI];
      SVal SubExprVal = State->getSVal(SubExpr, LCtx);
      ValuesToInvalidate.push_back(SubExprVal);
    }

    State = State->invalidateRegions(ValuesToInvalidate, AE,
                                    currBldrCtx->blockCount(),
                                    LCtx,
                                    /*CausedByPointerEscape*/true,
                                    /*Symbols=*/nullptr);

    SVal ResultVal = UnknownVal();
    State = State->BindExpr(AE, LCtx, ResultVal);
    Bldr.generateNode(AE, *I, State, nullptr,
                      ProgramPoint::PostStmtKind);
  }

  getCheckerManager().runCheckersForPostStmt(Dst, AfterInvalidateSet, AE, *this);
}

namespace {
class CollectReachableSymbolsCallback final : public SymbolVisitor {
  InvalidatedSymbols Symbols;

public:
  CollectReachableSymbolsCallback(ProgramStateRef State) {}
  const InvalidatedSymbols &getSymbols() const { return Symbols; }

  bool VisitSymbol(SymbolRef Sym) override {
    Symbols.insert(Sym);
    return true;
  }
};
} // end anonymous namespace

// A value escapes in three possible cases:
// (1) We are binding to something that is not a memory region.
// (2) We are binding to a MemrRegion that does not have stack storage.
// (3) We are binding to a MemRegion with stack storage that the store
//     does not understand.
ProgramStateRef ExprEngine::processPointerEscapedOnBind(ProgramStateRef State,
                                                        SVal Loc, SVal Val) {
  // Are we storing to something that causes the value to "escape"?
  bool escapes = true;

  // TODO: Move to StoreManager.
  if (Optional<loc::MemRegionVal> regionLoc = Loc.getAs<loc::MemRegionVal>()) {
    escapes = !regionLoc->getRegion()->hasStackStorage();

    if (!escapes) {
      // To test (3), generate a new state with the binding added.  If it is
      // the same state, then it escapes (since the store cannot represent
      // the binding).
      // Do this only if we know that the store is not supposed to generate the
      // same state.
      SVal StoredVal = State->getSVal(regionLoc->getRegion());
      if (StoredVal != Val)
        escapes = (State == (State->bindLoc(*regionLoc, Val)));
    }
  }

  // If our store can represent the binding and we aren't storing to something
  // that doesn't have local storage then just return and have the simulation
  // state continue as is.
  if (!escapes)
    return State;

  // Otherwise, find all symbols referenced by 'val' that we are tracking
  // and stop tracking them.
  CollectReachableSymbolsCallback Scanner =
      State->scanReachableSymbols<CollectReachableSymbolsCallback>(Val);
  const InvalidatedSymbols &EscapedSymbols = Scanner.getSymbols();
  State = getCheckerManager().runCheckersForPointerEscape(State,
                                                          EscapedSymbols,
                                                          /*CallEvent*/ nullptr,
                                                          PSK_EscapeOnBind,
                                                          nullptr);

  return State;
}

ProgramStateRef
ExprEngine::notifyCheckersOfPointerEscape(ProgramStateRef State,
    const InvalidatedSymbols *Invalidated,
    ArrayRef<const MemRegion *> ExplicitRegions,
    ArrayRef<const MemRegion *> Regions,
    const CallEvent *Call,
    RegionAndSymbolInvalidationTraits &ITraits) {

  if (!Invalidated || Invalidated->empty())
    return State;

  if (!Call)
    return getCheckerManager().runCheckersForPointerEscape(State,
                                                           *Invalidated,
                                                           nullptr,
                                                           PSK_EscapeOther,
                                                           &ITraits);

  // If the symbols were invalidated by a call, we want to find out which ones
  // were invalidated directly due to being arguments to the call.
  InvalidatedSymbols SymbolsDirectlyInvalidated;
  for (ArrayRef<const MemRegion *>::iterator I = ExplicitRegions.begin(),
      E = ExplicitRegions.end(); I != E; ++I) {
    if (const SymbolicRegion *R = (*I)->StripCasts()->getAs<SymbolicRegion>())
      SymbolsDirectlyInvalidated.insert(R->getSymbol());
  }

  InvalidatedSymbols SymbolsIndirectlyInvalidated;
  for (InvalidatedSymbols::const_iterator I=Invalidated->begin(),
      E = Invalidated->end(); I!=E; ++I) {
    SymbolRef sym = *I;
    if (SymbolsDirectlyInvalidated.count(sym))
      continue;
    SymbolsIndirectlyInvalidated.insert(sym);
  }

  if (!SymbolsDirectlyInvalidated.empty())
    State = getCheckerManager().runCheckersForPointerEscape(State,
        SymbolsDirectlyInvalidated, Call, PSK_DirectEscapeOnCall, &ITraits);

  // Notify about the symbols that get indirectly invalidated by the call.
  if (!SymbolsIndirectlyInvalidated.empty())
    State = getCheckerManager().runCheckersForPointerEscape(State,
        SymbolsIndirectlyInvalidated, Call, PSK_IndirectEscapeOnCall, &ITraits);

  return State;
}

/// evalBind - Handle the semantics of binding a value to a specific location.
///  This method is used by evalStore and (soon) VisitDeclStmt, and others.
void ExprEngine::evalBind(ExplodedNodeSet &Dst, const Stmt *StoreE,
                          ExplodedNode *Pred,
                          SVal location, SVal Val,
                          bool atDeclInit, const ProgramPoint *PP) {

  const LocationContext *LC = Pred->getLocationContext();
  PostStmt PS(StoreE, LC);
  if (!PP)
    PP = &PS;

  // Do a previsit of the bind.
  ExplodedNodeSet CheckedSet;
  getCheckerManager().runCheckersForBind(CheckedSet, Pred, location, Val,
                                         StoreE, *this, *PP);

  StmtNodeBuilder Bldr(CheckedSet, Dst, *currBldrCtx);

  // If the location is not a 'Loc', it will already be handled by
  // the checkers.  There is nothing left to do.
  if (!location.getAs<Loc>()) {
    const ProgramPoint L = PostStore(StoreE, LC, /*Loc*/nullptr,
                                     /*tag*/nullptr);
    ProgramStateRef state = Pred->getState();
    state = processPointerEscapedOnBind(state, location, Val);
    Bldr.generateNode(L, state, Pred);
    return;
  }

  for (ExplodedNodeSet::iterator I = CheckedSet.begin(), E = CheckedSet.end();
       I!=E; ++I) {
    ExplodedNode *PredI = *I;
    ProgramStateRef state = PredI->getState();

    state = processPointerEscapedOnBind(state, location, Val);

    // When binding the value, pass on the hint that this is a initialization.
    // For initializations, we do not need to inform clients of region
    // changes.
    state = state->bindLoc(location.castAs<Loc>(),
                           Val, /* notifyChanges = */ !atDeclInit);

    const MemRegion *LocReg = nullptr;
    if (Optional<loc::MemRegionVal> LocRegVal =
            location.getAs<loc::MemRegionVal>()) {
      LocReg = LocRegVal->getRegion();
    }

    const ProgramPoint L = PostStore(StoreE, LC, LocReg, nullptr);
    Bldr.generateNode(L, state, PredI);
  }
}

/// evalStore - Handle the semantics of a store via an assignment.
///  @param Dst The node set to store generated state nodes
///  @param AssignE The assignment expression if the store happens in an
///         assignment.
///  @param LocationE The location expression that is stored to.
///  @param state The current simulation state
///  @param location The location to store the value
///  @param Val The value to be stored
void ExprEngine::evalStore(ExplodedNodeSet &Dst, const Expr *AssignE,
                             const Expr *LocationE,
                             ExplodedNode *Pred,
                             ProgramStateRef state, SVal location, SVal Val,
                             const ProgramPointTag *tag) {
  // Proceed with the store.  We use AssignE as the anchor for the PostStore
  // ProgramPoint if it is non-NULL, and LocationE otherwise.
  const Expr *StoreE = AssignE ? AssignE : LocationE;

  // Evaluate the location (checks for bad dereferences).
  ExplodedNodeSet Tmp;
  evalLocation(Tmp, AssignE, LocationE, Pred, state, location, tag, false);

  if (Tmp.empty())
    return;

  if (location.isUndef())
    return;

  for (ExplodedNodeSet::iterator NI=Tmp.begin(), NE=Tmp.end(); NI!=NE; ++NI)
    evalBind(Dst, StoreE, *NI, location, Val, false);
}

void ExprEngine::evalLoad(ExplodedNodeSet &Dst,
                          const Expr *NodeEx,
                          const Expr *BoundEx,
                          ExplodedNode *Pred,
                          ProgramStateRef state,
                          SVal location,
                          const ProgramPointTag *tag,
                          QualType LoadTy)
{
  assert(!location.getAs<NonLoc>() && "location cannot be a NonLoc.");

  // Are we loading from a region?  This actually results in two loads; one
  // to fetch the address of the referenced value and one to fetch the
  // referenced value.
  if (const TypedValueRegion *TR =
        dyn_cast_or_null<TypedValueRegion>(location.getAsRegion())) {

    QualType ValTy = TR->getValueType();
    if (const ReferenceType *RT = ValTy->getAs<ReferenceType>()) {
      static SimpleProgramPointTag
             loadReferenceTag(TagProviderName, "Load Reference");
      ExplodedNodeSet Tmp;
      evalLoadCommon(Tmp, NodeEx, BoundEx, Pred, state,
                     location, &loadReferenceTag,
                     getContext().getPointerType(RT->getPointeeType()));

      // Perform the load from the referenced value.
      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end() ; I!=E; ++I) {
        state = (*I)->getState();
        location = state->getSVal(BoundEx, (*I)->getLocationContext());
        evalLoadCommon(Dst, NodeEx, BoundEx, *I, state, location, tag, LoadTy);
      }
      return;
    }
  }

  evalLoadCommon(Dst, NodeEx, BoundEx, Pred, state, location, tag, LoadTy);
}

void ExprEngine::evalLoadCommon(ExplodedNodeSet &Dst,
                                const Expr *NodeEx,
                                const Expr *BoundEx,
                                ExplodedNode *Pred,
                                ProgramStateRef state,
                                SVal location,
                                const ProgramPointTag *tag,
                                QualType LoadTy) {
  assert(NodeEx);
  assert(BoundEx);
  // Evaluate the location (checks for bad dereferences).
  ExplodedNodeSet Tmp;
  evalLocation(Tmp, NodeEx, BoundEx, Pred, state, location, tag, true);
  if (Tmp.empty())
    return;

  StmtNodeBuilder Bldr(Tmp, Dst, *currBldrCtx);
  if (location.isUndef())
    return;

  // Proceed with the load.
  for (ExplodedNodeSet::iterator NI=Tmp.begin(), NE=Tmp.end(); NI!=NE; ++NI) {
    state = (*NI)->getState();
    const LocationContext *LCtx = (*NI)->getLocationContext();

    SVal V = UnknownVal();
    if (location.isValid()) {
      if (LoadTy.isNull())
        LoadTy = BoundEx->getType();
      V = state->getSVal(location.castAs<Loc>(), LoadTy);
    }

    Bldr.generateNode(NodeEx, *NI, state->BindExpr(BoundEx, LCtx, V), tag,
                      ProgramPoint::PostLoadKind);
  }
}

void ExprEngine::evalLocation(ExplodedNodeSet &Dst,
                              const Stmt *NodeEx,
                              const Stmt *BoundEx,
                              ExplodedNode *Pred,
                              ProgramStateRef state,
                              SVal location,
                              const ProgramPointTag *tag,
                              bool isLoad) {
  StmtNodeBuilder BldrTop(Pred, Dst, *currBldrCtx);
  // Early checks for performance reason.
  if (location.isUnknown()) {
    return;
  }

  ExplodedNodeSet Src;
  BldrTop.takeNodes(Pred);
  StmtNodeBuilder Bldr(Pred, Src, *currBldrCtx);
  if (Pred->getState() != state) {
    // Associate this new state with an ExplodedNode.
    // FIXME: If I pass null tag, the graph is incorrect, e.g for
    //   int *p;
    //   p = 0;
    //   *p = 0xDEADBEEF;
    // "p = 0" is not noted as "Null pointer value stored to 'p'" but
    // instead "int *p" is noted as
    // "Variable 'p' initialized to a null pointer value"

    static SimpleProgramPointTag tag(TagProviderName, "Location");
    Bldr.generateNode(NodeEx, Pred, state, &tag);
  }
  ExplodedNodeSet Tmp;
  getCheckerManager().runCheckersForLocation(Tmp, Src, location, isLoad,
                                             NodeEx, BoundEx, *this);
  BldrTop.addNodes(Tmp);
}

std::pair<const ProgramPointTag *, const ProgramPointTag*>
ExprEngine::geteagerlyAssumeBinOpBifurcationTags() {
  static SimpleProgramPointTag
         eagerlyAssumeBinOpBifurcationTrue(TagProviderName,
                                           "Eagerly Assume True"),
         eagerlyAssumeBinOpBifurcationFalse(TagProviderName,
                                            "Eagerly Assume False");
  return std::make_pair(&eagerlyAssumeBinOpBifurcationTrue,
                        &eagerlyAssumeBinOpBifurcationFalse);
}

void ExprEngine::evalEagerlyAssumeBinOpBifurcation(ExplodedNodeSet &Dst,
                                                   ExplodedNodeSet &Src,
                                                   const Expr *Ex) {
  StmtNodeBuilder Bldr(Src, Dst, *currBldrCtx);

  for (ExplodedNodeSet::iterator I=Src.begin(), E=Src.end(); I!=E; ++I) {
    ExplodedNode *Pred = *I;
    // Test if the previous node was as the same expression.  This can happen
    // when the expression fails to evaluate to anything meaningful and
    // (as an optimization) we don't generate a node.
    ProgramPoint P = Pred->getLocation();
    if (!P.getAs<PostStmt>() || P.castAs<PostStmt>().getStmt() != Ex) {
      continue;
    }

    ProgramStateRef state = Pred->getState();
    SVal V = state->getSVal(Ex, Pred->getLocationContext());
    Optional<nonloc::SymbolVal> SEV = V.getAs<nonloc::SymbolVal>();
    if (SEV && SEV->isExpression()) {
      const std::pair<const ProgramPointTag *, const ProgramPointTag*> &tags =
        geteagerlyAssumeBinOpBifurcationTags();

      ProgramStateRef StateTrue, StateFalse;
      std::tie(StateTrue, StateFalse) = state->assume(*SEV);

      // First assume that the condition is true.
      if (StateTrue) {
        SVal Val = svalBuilder.makeIntVal(1U, Ex->getType());
        StateTrue = StateTrue->BindExpr(Ex, Pred->getLocationContext(), Val);
        Bldr.generateNode(Ex, Pred, StateTrue, tags.first);
      }

      // Next, assume that the condition is false.
      if (StateFalse) {
        SVal Val = svalBuilder.makeIntVal(0U, Ex->getType());
        StateFalse = StateFalse->BindExpr(Ex, Pred->getLocationContext(), Val);
        Bldr.generateNode(Ex, Pred, StateFalse, tags.second);
      }
    }
  }
}

void ExprEngine::VisitGCCAsmStmt(const GCCAsmStmt *A, ExplodedNode *Pred,
                                 ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);
  // We have processed both the inputs and the outputs.  All of the outputs
  // should evaluate to Locs.  Nuke all of their values.

  // FIXME: Some day in the future it would be nice to allow a "plug-in"
  // which interprets the inline asm and stores proper results in the
  // outputs.

  ProgramStateRef state = Pred->getState();

  for (const Expr *O : A->outputs()) {
    SVal X = state->getSVal(O, Pred->getLocationContext());
    assert (!X.getAs<NonLoc>());  // Should be an Lval, or unknown, undef.

    if (Optional<Loc> LV = X.getAs<Loc>())
      state = state->bindLoc(*LV, UnknownVal());
  }

  Bldr.generateNode(A, Pred, state);
}

void ExprEngine::VisitMSAsmStmt(const MSAsmStmt *A, ExplodedNode *Pred,
                                ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);
  Bldr.generateNode(A, Pred, Pred->getState());
}

//===----------------------------------------------------------------------===//
// Visualization.
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
static ExprEngine* GraphPrintCheckerState;
static SourceManager* GraphPrintSourceManager;

namespace llvm {
template<>
struct DOTGraphTraits<ExplodedNode*> :
  public DefaultDOTGraphTraits {

  DOTGraphTraits (bool isSimple=false) : DefaultDOTGraphTraits(isSimple) {}

  // FIXME: Since we do not cache error nodes in ExprEngine now, this does not
  // work.
  static std::string getNodeAttributes(const ExplodedNode *N, void*) {
    return "";
  }

  // De-duplicate some source location pretty-printing.
  static void printLocation(raw_ostream &Out, SourceLocation SLoc) {
    if (SLoc.isFileID()) {
      Out << "\\lline="
        << GraphPrintSourceManager->getExpansionLineNumber(SLoc)
        << " col="
        << GraphPrintSourceManager->getExpansionColumnNumber(SLoc)
        << "\\l";
    }
  }
  static void printLocation2(raw_ostream &Out, SourceLocation SLoc) {
    if (SLoc.isFileID() && GraphPrintSourceManager->isInMainFile(SLoc))
      Out << "line " << GraphPrintSourceManager->getExpansionLineNumber(SLoc);
    else
      SLoc.print(Out, *GraphPrintSourceManager);
  }

  static std::string getNodeLabel(const ExplodedNode *N, void*){

    std::string sbuf;
    llvm::raw_string_ostream Out(sbuf);

    // Program Location.
    ProgramPoint Loc = N->getLocation();

    switch (Loc.getKind()) {
      case ProgramPoint::BlockEntranceKind: {
        Out << "Block Entrance: B"
            << Loc.castAs<BlockEntrance>().getBlock()->getBlockID();
        break;
      }

      case ProgramPoint::BlockExitKind:
        assert (false);
        break;

      case ProgramPoint::CallEnterKind:
        Out << "CallEnter";
        break;

      case ProgramPoint::CallExitBeginKind:
        Out << "CallExitBegin";
        break;

      case ProgramPoint::CallExitEndKind:
        Out << "CallExitEnd";
        break;

      case ProgramPoint::PostStmtPurgeDeadSymbolsKind:
        Out << "PostStmtPurgeDeadSymbols";
        break;

      case ProgramPoint::PreStmtPurgeDeadSymbolsKind:
        Out << "PreStmtPurgeDeadSymbols";
        break;

      case ProgramPoint::EpsilonKind:
        Out << "Epsilon Point";
        break;

      case ProgramPoint::PreImplicitCallKind: {
        ImplicitCallPoint PC = Loc.castAs<ImplicitCallPoint>();
        Out << "PreCall: ";

        // FIXME: Get proper printing options.
        PC.getDecl()->print(Out, LangOptions());
        printLocation(Out, PC.getLocation());
        break;
      }

      case ProgramPoint::PostImplicitCallKind: {
        ImplicitCallPoint PC = Loc.castAs<ImplicitCallPoint>();
        Out << "PostCall: ";

        // FIXME: Get proper printing options.
        PC.getDecl()->print(Out, LangOptions());
        printLocation(Out, PC.getLocation());
        break;
      }

      case ProgramPoint::PostInitializerKind: {
        Out << "PostInitializer: ";
        const CXXCtorInitializer *Init =
          Loc.castAs<PostInitializer>().getInitializer();
        if (const FieldDecl *FD = Init->getAnyMember())
          Out << *FD;
        else {
          QualType Ty = Init->getTypeSourceInfo()->getType();
          Ty = Ty.getLocalUnqualifiedType();
          LangOptions LO; // FIXME.
          Ty.print(Out, LO);
        }
        break;
      }

      case ProgramPoint::BlockEdgeKind: {
        const BlockEdge &E = Loc.castAs<BlockEdge>();
        Out << "Edge: (B" << E.getSrc()->getBlockID() << ", B"
            << E.getDst()->getBlockID()  << ')';

        if (const Stmt *T = E.getSrc()->getTerminator()) {
          SourceLocation SLoc = T->getLocStart();

          Out << "\\|Terminator: ";
          LangOptions LO; // FIXME.
          E.getSrc()->printTerminator(Out, LO);

          if (SLoc.isFileID()) {
            Out << "\\lline="
              << GraphPrintSourceManager->getExpansionLineNumber(SLoc)
              << " col="
              << GraphPrintSourceManager->getExpansionColumnNumber(SLoc);
          }

          if (isa<SwitchStmt>(T)) {
            const Stmt *Label = E.getDst()->getLabel();

            if (Label) {
              if (const CaseStmt *C = dyn_cast<CaseStmt>(Label)) {
                Out << "\\lcase ";
                LangOptions LO; // FIXME.
                if (C->getLHS())
                  C->getLHS()->printPretty(Out, nullptr, PrintingPolicy(LO));

                if (const Stmt *RHS = C->getRHS()) {
                  Out << " .. ";
                  RHS->printPretty(Out, nullptr, PrintingPolicy(LO));
                }

                Out << ":";
              }
              else {
                assert (isa<DefaultStmt>(Label));
                Out << "\\ldefault:";
              }
            }
            else
              Out << "\\l(implicit) default:";
          }
          else if (isa<IndirectGotoStmt>(T)) {
            // FIXME
          }
          else {
            Out << "\\lCondition: ";
            if (*E.getSrc()->succ_begin() == E.getDst())
              Out << "true";
            else
              Out << "false";
          }

          Out << "\\l";
        }

        break;
      }

      default: {
        const Stmt *S = Loc.castAs<StmtPoint>().getStmt();
        assert(S != nullptr && "Expecting non-null Stmt");

        Out << S->getStmtClassName() << ' ' << (const void*) S << ' ';
        LangOptions LO; // FIXME.
        S->printPretty(Out, nullptr, PrintingPolicy(LO));
        printLocation(Out, S->getLocStart());

        if (Loc.getAs<PreStmt>())
          Out << "\\lPreStmt\\l;";
        else if (Loc.getAs<PostLoad>())
          Out << "\\lPostLoad\\l;";
        else if (Loc.getAs<PostStore>())
          Out << "\\lPostStore\\l";
        else if (Loc.getAs<PostLValue>())
          Out << "\\lPostLValue\\l";

        break;
      }
    }

    ProgramStateRef state = N->getState();
    Out << "\\|StateID: " << (const void*) state.get()
        << " NodeID: " << (const void*) N << "\\|";

    // Analysis stack backtrace.
    Out << "Location context stack (from current to outer):\\l";
    const LocationContext *LC = Loc.getLocationContext();
    unsigned Idx = 0;
    for (; LC; LC = LC->getParent(), ++Idx) {
      Out << Idx << ". (" << (const void *)LC << ") ";
      switch (LC->getKind()) {
      case LocationContext::StackFrame:
        if (const NamedDecl *D = dyn_cast<NamedDecl>(LC->getDecl()))
          Out << "Calling " << D->getQualifiedNameAsString();
        else
          Out << "Calling anonymous code";
        if (const Stmt *S = cast<StackFrameContext>(LC)->getCallSite()) {
          Out << " at ";
          printLocation2(Out, S->getLocStart());
        }
        break;
      case LocationContext::Block:
        Out << "Invoking block";
        if (const Decl *D = cast<BlockInvocationContext>(LC)->getBlockDecl()) {
          Out << " defined at ";
          printLocation2(Out, D->getLocStart());
        }
        break;
      case LocationContext::Scope:
        Out << "Entering scope";
        // FIXME: Add more info once ScopeContext is activated.
        break;
      }
      Out << "\\l";
    }
    Out << "\\l";

    state->printDOT(Out);

    Out << "\\l";

    if (const ProgramPointTag *tag = Loc.getTag()) {
      Out << "\\|Tag: " << tag->getTagDescription();
      Out << "\\l";
    }
    return Out.str();
  }
};
} // end llvm namespace
#endif

void ExprEngine::ViewGraph(bool trim) {
#ifndef NDEBUG
  if (trim) {
    std::vector<const ExplodedNode*> Src;

    // Flush any outstanding reports to make sure we cover all the nodes.
    // This does not cause them to get displayed.
    for (BugReporter::iterator I=BR.begin(), E=BR.end(); I!=E; ++I)
      const_cast<BugType*>(*I)->FlushReports(BR);

    // Iterate through the reports and get their nodes.
    for (BugReporter::EQClasses_iterator
           EI = BR.EQClasses_begin(), EE = BR.EQClasses_end(); EI != EE; ++EI) {
      ExplodedNode *N = const_cast<ExplodedNode*>(EI->begin()->getErrorNode());
      if (N) Src.push_back(N);
    }

    ViewGraph(Src);
  }
  else {
    GraphPrintCheckerState = this;
    GraphPrintSourceManager = &getContext().getSourceManager();

    llvm::ViewGraph(*G.roots_begin(), "ExprEngine");

    GraphPrintCheckerState = nullptr;
    GraphPrintSourceManager = nullptr;
  }
#endif
}

void ExprEngine::ViewGraph(ArrayRef<const ExplodedNode*> Nodes) {
#ifndef NDEBUG
  GraphPrintCheckerState = this;
  GraphPrintSourceManager = &getContext().getSourceManager();

  std::unique_ptr<ExplodedGraph> TrimmedG(G.trim(Nodes));

  if (!TrimmedG.get())
    llvm::errs() << "warning: Trimmed ExplodedGraph is empty.\n";
  else
    llvm::ViewGraph(*TrimmedG->roots_begin(), "TrimmedExprEngine");

  GraphPrintCheckerState = nullptr;
  GraphPrintSourceManager = nullptr;
#endif
}
