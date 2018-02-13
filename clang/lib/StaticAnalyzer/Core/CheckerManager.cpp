//===--- CheckerManager.cpp - Static Analyzer Checker Manager -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the Static Analyzer Checker Manager.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/AST/DeclBase.h"
#include "clang/Analysis/ProgramPoint.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

bool CheckerManager::hasPathSensitiveCheckers() const {
  return !StmtCheckers.empty()              ||
         !PreObjCMessageCheckers.empty()    ||
         !PostObjCMessageCheckers.empty()   ||
         !PreCallCheckers.empty()    ||
         !PostCallCheckers.empty()   ||
         !LocationCheckers.empty()          ||
         !BindCheckers.empty()              ||
         !EndAnalysisCheckers.empty()       ||
         !EndFunctionCheckers.empty()           ||
         !BranchConditionCheckers.empty()   ||
         !LiveSymbolsCheckers.empty()       ||
         !DeadSymbolsCheckers.empty()       ||
         !RegionChangesCheckers.empty()     ||
         !EvalAssumeCheckers.empty()        ||
         !EvalCallCheckers.empty();
}

void CheckerManager::finishedCheckerRegistration() {
#ifndef NDEBUG
  // Make sure that for every event that has listeners, there is at least
  // one dispatcher registered for it.
  for (llvm::DenseMap<EventTag, EventInfo>::iterator
         I = Events.begin(), E = Events.end(); I != E; ++I)
    assert(I->second.HasDispatcher && "No dispatcher registered for an event");
#endif
}

//===----------------------------------------------------------------------===//
// Functions for running checkers for AST traversing..
//===----------------------------------------------------------------------===//

void CheckerManager::runCheckersOnASTDecl(const Decl *D, AnalysisManager& mgr,
                                          BugReporter &BR) {
  assert(D);

  unsigned DeclKind = D->getKind();
  CachedDeclCheckers *checkers = nullptr;
  CachedDeclCheckersMapTy::iterator CCI = CachedDeclCheckersMap.find(DeclKind);
  if (CCI != CachedDeclCheckersMap.end()) {
    checkers = &(CCI->second);
  } else {
    // Find the checkers that should run for this Decl and cache them.
    checkers = &CachedDeclCheckersMap[DeclKind];
    for (unsigned i = 0, e = DeclCheckers.size(); i != e; ++i) {
      DeclCheckerInfo &info = DeclCheckers[i];
      if (info.IsForDeclFn(D))
        checkers->push_back(info.CheckFn);
    }
  }

  assert(checkers);
  for (CachedDeclCheckers::iterator
         I = checkers->begin(), E = checkers->end(); I != E; ++I)
    (*I)(D, mgr, BR);
}

void CheckerManager::runCheckersOnASTBody(const Decl *D, AnalysisManager& mgr,
                                          BugReporter &BR) {
  assert(D && D->hasBody());

  for (unsigned i = 0, e = BodyCheckers.size(); i != e; ++i)
    BodyCheckers[i](D, mgr, BR);
}

//===----------------------------------------------------------------------===//
// Functions for running checkers for path-sensitive checking.
//===----------------------------------------------------------------------===//

template <typename CHECK_CTX>
static void expandGraphWithCheckers(CHECK_CTX checkCtx,
                                    ExplodedNodeSet &Dst,
                                    const ExplodedNodeSet &Src) {
  const NodeBuilderContext &BldrCtx = checkCtx.Eng.getBuilderContext();
  if (Src.empty())
    return;

  typename CHECK_CTX::CheckersTy::const_iterator
      I = checkCtx.checkers_begin(), E = checkCtx.checkers_end();
  if (I == E) {
    Dst.insert(Src);
    return;
  }

  ExplodedNodeSet Tmp1, Tmp2;
  const ExplodedNodeSet *PrevSet = &Src;

  for (; I != E; ++I) {
    ExplodedNodeSet *CurrSet = nullptr;
    if (I+1 == E)
      CurrSet = &Dst;
    else {
      CurrSet = (PrevSet == &Tmp1) ? &Tmp2 : &Tmp1;
      CurrSet->clear();
    }

    NodeBuilder B(*PrevSet, *CurrSet, BldrCtx);
    for (ExplodedNodeSet::iterator NI = PrevSet->begin(), NE = PrevSet->end();
         NI != NE; ++NI) {
      checkCtx.runChecker(*I, B, *NI);
    }

    // If all the produced transitions are sinks, stop.
    if (CurrSet->empty())
      return;

    // Update which NodeSet is the current one.
    PrevSet = CurrSet;
  }
}

namespace {
  struct CheckStmtContext {
    typedef SmallVectorImpl<CheckerManager::CheckStmtFunc> CheckersTy;
    bool IsPreVisit;
    const CheckersTy &Checkers;
    const Stmt *S;
    ExprEngine &Eng;
    bool WasInlined;

    CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
    CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

    CheckStmtContext(bool isPreVisit, const CheckersTy &checkers,
                     const Stmt *s, ExprEngine &eng, bool wasInlined = false)
      : IsPreVisit(isPreVisit), Checkers(checkers), S(s), Eng(eng),
        WasInlined(wasInlined) {}

    void runChecker(CheckerManager::CheckStmtFunc checkFn,
                    NodeBuilder &Bldr, ExplodedNode *Pred) {
      // FIXME: Remove respondsToCallback from CheckerContext;
      ProgramPoint::Kind K =  IsPreVisit ? ProgramPoint::PreStmtKind :
                                           ProgramPoint::PostStmtKind;
      const ProgramPoint &L = ProgramPoint::getProgramPoint(S, K,
                                Pred->getLocationContext(), checkFn.Checker);
      CheckerContext C(Bldr, Eng, Pred, L, WasInlined);
      checkFn(S, C);
    }
  };
}

/// \brief Run checkers for visiting Stmts.
void CheckerManager::runCheckersForStmt(bool isPreVisit,
                                        ExplodedNodeSet &Dst,
                                        const ExplodedNodeSet &Src,
                                        const Stmt *S,
                                        ExprEngine &Eng,
                                        bool WasInlined) {
  CheckStmtContext C(isPreVisit, getCachedStmtCheckersFor(S, isPreVisit),
                     S, Eng, WasInlined);
  expandGraphWithCheckers(C, Dst, Src);
}

namespace {
  struct CheckObjCMessageContext {
    typedef std::vector<CheckerManager::CheckObjCMessageFunc> CheckersTy;

    ObjCMessageVisitKind Kind;
    bool WasInlined;
    const CheckersTy &Checkers;
    const ObjCMethodCall &Msg;
    ExprEngine &Eng;

    CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
    CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

    CheckObjCMessageContext(ObjCMessageVisitKind visitKind,
                            const CheckersTy &checkers,
                            const ObjCMethodCall &msg, ExprEngine &eng,
                            bool wasInlined)
      : Kind(visitKind), WasInlined(wasInlined), Checkers(checkers),
        Msg(msg), Eng(eng) { }

    void runChecker(CheckerManager::CheckObjCMessageFunc checkFn,
                    NodeBuilder &Bldr, ExplodedNode *Pred) {

      bool IsPreVisit;

      switch (Kind) {
        case ObjCMessageVisitKind::Pre:
          IsPreVisit = true;
          break;
        case ObjCMessageVisitKind::MessageNil:
        case ObjCMessageVisitKind::Post:
          IsPreVisit = false;
          break;
      }

      const ProgramPoint &L = Msg.getProgramPoint(IsPreVisit,checkFn.Checker);
      CheckerContext C(Bldr, Eng, Pred, L, WasInlined);

      checkFn(*Msg.cloneWithState<ObjCMethodCall>(Pred->getState()), C);
    }
  };
}

/// \brief Run checkers for visiting obj-c messages.
void CheckerManager::runCheckersForObjCMessage(ObjCMessageVisitKind visitKind,
                                               ExplodedNodeSet &Dst,
                                               const ExplodedNodeSet &Src,
                                               const ObjCMethodCall &msg,
                                               ExprEngine &Eng,
                                               bool WasInlined) {
  auto &checkers = getObjCMessageCheckers(visitKind);
  CheckObjCMessageContext C(visitKind, checkers, msg, Eng, WasInlined);
  expandGraphWithCheckers(C, Dst, Src);
}

const std::vector<CheckerManager::CheckObjCMessageFunc> &
CheckerManager::getObjCMessageCheckers(ObjCMessageVisitKind Kind) {
  switch (Kind) {
  case ObjCMessageVisitKind::Pre:
    return PreObjCMessageCheckers;
    break;
  case ObjCMessageVisitKind::Post:
    return PostObjCMessageCheckers;
  case ObjCMessageVisitKind::MessageNil:
    return ObjCMessageNilCheckers;
  }
  llvm_unreachable("Unknown Kind");
}
namespace {
  // FIXME: This has all the same signatures as CheckObjCMessageContext.
  // Is there a way we can merge the two?
  struct CheckCallContext {
    typedef std::vector<CheckerManager::CheckCallFunc> CheckersTy;
    bool IsPreVisit, WasInlined;
    const CheckersTy &Checkers;
    const CallEvent &Call;
    ExprEngine &Eng;

    CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
    CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

    CheckCallContext(bool isPreVisit, const CheckersTy &checkers,
                     const CallEvent &call, ExprEngine &eng,
                     bool wasInlined)
    : IsPreVisit(isPreVisit), WasInlined(wasInlined), Checkers(checkers),
      Call(call), Eng(eng) { }

    void runChecker(CheckerManager::CheckCallFunc checkFn,
                    NodeBuilder &Bldr, ExplodedNode *Pred) {
      const ProgramPoint &L = Call.getProgramPoint(IsPreVisit,checkFn.Checker);
      CheckerContext C(Bldr, Eng, Pred, L, WasInlined);

      checkFn(*Call.cloneWithState(Pred->getState()), C);
    }
  };
}

/// \brief Run checkers for visiting an abstract call event.
void CheckerManager::runCheckersForCallEvent(bool isPreVisit,
                                             ExplodedNodeSet &Dst,
                                             const ExplodedNodeSet &Src,
                                             const CallEvent &Call,
                                             ExprEngine &Eng,
                                             bool WasInlined) {
  CheckCallContext C(isPreVisit,
                     isPreVisit ? PreCallCheckers
                                : PostCallCheckers,
                     Call, Eng, WasInlined);
  expandGraphWithCheckers(C, Dst, Src);
}

namespace {
  struct CheckLocationContext {
    typedef std::vector<CheckerManager::CheckLocationFunc> CheckersTy;
    const CheckersTy &Checkers;
    SVal Loc;
    bool IsLoad;
    const Stmt *NodeEx; /* Will become a CFGStmt */
    const Stmt *BoundEx;
    ExprEngine &Eng;

    CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
    CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

    CheckLocationContext(const CheckersTy &checkers,
                         SVal loc, bool isLoad, const Stmt *NodeEx,
                         const Stmt *BoundEx,
                         ExprEngine &eng)
      : Checkers(checkers), Loc(loc), IsLoad(isLoad), NodeEx(NodeEx),
        BoundEx(BoundEx), Eng(eng) {}

    void runChecker(CheckerManager::CheckLocationFunc checkFn,
                    NodeBuilder &Bldr, ExplodedNode *Pred) {
      ProgramPoint::Kind K =  IsLoad ? ProgramPoint::PreLoadKind :
                                       ProgramPoint::PreStoreKind;
      const ProgramPoint &L =
        ProgramPoint::getProgramPoint(NodeEx, K,
                                      Pred->getLocationContext(),
                                      checkFn.Checker);
      CheckerContext C(Bldr, Eng, Pred, L);
      checkFn(Loc, IsLoad, BoundEx, C);
    }
  };
}

/// \brief Run checkers for load/store of a location.

void CheckerManager::runCheckersForLocation(ExplodedNodeSet &Dst,
                                            const ExplodedNodeSet &Src,
                                            SVal location, bool isLoad,
                                            const Stmt *NodeEx,
                                            const Stmt *BoundEx,
                                            ExprEngine &Eng) {
  CheckLocationContext C(LocationCheckers, location, isLoad, NodeEx,
                         BoundEx, Eng);
  expandGraphWithCheckers(C, Dst, Src);
}

namespace {
  struct CheckBindContext {
    typedef std::vector<CheckerManager::CheckBindFunc> CheckersTy;
    const CheckersTy &Checkers;
    SVal Loc;
    SVal Val;
    const Stmt *S;
    ExprEngine &Eng;
    const ProgramPoint &PP;

    CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
    CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

    CheckBindContext(const CheckersTy &checkers,
                     SVal loc, SVal val, const Stmt *s, ExprEngine &eng,
                     const ProgramPoint &pp)
      : Checkers(checkers), Loc(loc), Val(val), S(s), Eng(eng), PP(pp) {}

    void runChecker(CheckerManager::CheckBindFunc checkFn,
                    NodeBuilder &Bldr, ExplodedNode *Pred) {
      const ProgramPoint &L = PP.withTag(checkFn.Checker);
      CheckerContext C(Bldr, Eng, Pred, L);

      checkFn(Loc, Val, S, C);
    }
  };
}

/// \brief Run checkers for binding of a value to a location.
void CheckerManager::runCheckersForBind(ExplodedNodeSet &Dst,
                                        const ExplodedNodeSet &Src,
                                        SVal location, SVal val,
                                        const Stmt *S, ExprEngine &Eng,
                                        const ProgramPoint &PP) {
  CheckBindContext C(BindCheckers, location, val, S, Eng, PP);
  expandGraphWithCheckers(C, Dst, Src);
}

void CheckerManager::runCheckersForEndAnalysis(ExplodedGraph &G,
                                               BugReporter &BR,
                                               ExprEngine &Eng) {
  for (unsigned i = 0, e = EndAnalysisCheckers.size(); i != e; ++i)
    EndAnalysisCheckers[i](G, BR, Eng);
}

namespace {
struct CheckBeginFunctionContext {
  typedef std::vector<CheckerManager::CheckBeginFunctionFunc> CheckersTy;
  const CheckersTy &Checkers;
  ExprEngine &Eng;
  const ProgramPoint &PP;

  CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
  CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

  CheckBeginFunctionContext(const CheckersTy &Checkers, ExprEngine &Eng,
                            const ProgramPoint &PP)
      : Checkers(Checkers), Eng(Eng), PP(PP) {}

  void runChecker(CheckerManager::CheckBeginFunctionFunc checkFn,
                  NodeBuilder &Bldr, ExplodedNode *Pred) {
    const ProgramPoint &L = PP.withTag(checkFn.Checker);
    CheckerContext C(Bldr, Eng, Pred, L);

    checkFn(C);
  }
};
}

void CheckerManager::runCheckersForBeginFunction(ExplodedNodeSet &Dst,
                                                 const BlockEdge &L,
                                                 ExplodedNode *Pred,
                                                 ExprEngine &Eng) {
  ExplodedNodeSet Src;
  Src.insert(Pred);
  CheckBeginFunctionContext C(BeginFunctionCheckers, Eng, L);
  expandGraphWithCheckers(C, Dst, Src);
}

/// \brief Run checkers for end of path.
// Note, We do not chain the checker output (like in expandGraphWithCheckers)
// for this callback since end of path nodes are expected to be final.
void CheckerManager::runCheckersForEndFunction(NodeBuilderContext &BC,
                                               ExplodedNodeSet &Dst,
                                               ExplodedNode *Pred,
                                               ExprEngine &Eng) {

  // We define the builder outside of the loop bacause if at least one checkers
  // creates a sucsessor for Pred, we do not need to generate an
  // autotransition for it.
  NodeBuilder Bldr(Pred, Dst, BC);
  for (unsigned i = 0, e = EndFunctionCheckers.size(); i != e; ++i) {
    CheckEndFunctionFunc checkFn = EndFunctionCheckers[i];

    const ProgramPoint &L = BlockEntrance(BC.Block,
                                          Pred->getLocationContext(),
                                          checkFn.Checker);
    CheckerContext C(Bldr, Eng, Pred, L);
    checkFn(C);
  }
}

namespace {
  struct CheckBranchConditionContext {
    typedef std::vector<CheckerManager::CheckBranchConditionFunc> CheckersTy;
    const CheckersTy &Checkers;
    const Stmt *Condition;
    ExprEngine &Eng;

    CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
    CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

    CheckBranchConditionContext(const CheckersTy &checkers,
                                const Stmt *Cond, ExprEngine &eng)
      : Checkers(checkers), Condition(Cond), Eng(eng) {}

    void runChecker(CheckerManager::CheckBranchConditionFunc checkFn,
                    NodeBuilder &Bldr, ExplodedNode *Pred) {
      ProgramPoint L = PostCondition(Condition, Pred->getLocationContext(),
                                     checkFn.Checker);
      CheckerContext C(Bldr, Eng, Pred, L);
      checkFn(Condition, C);
    }
  };
}

/// \brief Run checkers for branch condition.
void CheckerManager::runCheckersForBranchCondition(const Stmt *Condition,
                                                   ExplodedNodeSet &Dst,
                                                   ExplodedNode *Pred,
                                                   ExprEngine &Eng) {
  ExplodedNodeSet Src;
  Src.insert(Pred);
  CheckBranchConditionContext C(BranchConditionCheckers, Condition, Eng);
  expandGraphWithCheckers(C, Dst, Src);
}

namespace {
  struct CheckNewAllocatorContext {
    typedef std::vector<CheckerManager::CheckNewAllocatorFunc> CheckersTy;
    const CheckersTy &Checkers;
    const CXXNewExpr *NE;
    SVal Target;
    bool WasInlined;
    ExprEngine &Eng;

    CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
    CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

    CheckNewAllocatorContext(const CheckersTy &Checkers, const CXXNewExpr *NE,
                             SVal Target, bool WasInlined, ExprEngine &Eng)
        : Checkers(Checkers), NE(NE), Target(Target), WasInlined(WasInlined),
          Eng(Eng) {}

    void runChecker(CheckerManager::CheckNewAllocatorFunc checkFn,
                    NodeBuilder &Bldr, ExplodedNode *Pred) {
      ProgramPoint L = PostAllocatorCall(NE, Pred->getLocationContext());
      CheckerContext C(Bldr, Eng, Pred, L, WasInlined);
      checkFn(NE, Target, C);
    }
  };
}

void CheckerManager::runCheckersForNewAllocator(
    const CXXNewExpr *NE, SVal Target, ExplodedNodeSet &Dst, ExplodedNode *Pred,
    ExprEngine &Eng, bool WasInlined) {
  ExplodedNodeSet Src;
  Src.insert(Pred);
  CheckNewAllocatorContext C(NewAllocatorCheckers, NE, Target, WasInlined, Eng);
  expandGraphWithCheckers(C, Dst, Src);
}

/// \brief Run checkers for live symbols.
void CheckerManager::runCheckersForLiveSymbols(ProgramStateRef state,
                                               SymbolReaper &SymReaper) {
  for (unsigned i = 0, e = LiveSymbolsCheckers.size(); i != e; ++i)
    LiveSymbolsCheckers[i](state, SymReaper);
}

namespace {
  struct CheckDeadSymbolsContext {
    typedef std::vector<CheckerManager::CheckDeadSymbolsFunc> CheckersTy;
    const CheckersTy &Checkers;
    SymbolReaper &SR;
    const Stmt *S;
    ExprEngine &Eng;
    ProgramPoint::Kind ProgarmPointKind;

    CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
    CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

    CheckDeadSymbolsContext(const CheckersTy &checkers, SymbolReaper &sr,
                            const Stmt *s, ExprEngine &eng,
                            ProgramPoint::Kind K)
      : Checkers(checkers), SR(sr), S(s), Eng(eng), ProgarmPointKind(K) { }

    void runChecker(CheckerManager::CheckDeadSymbolsFunc checkFn,
                    NodeBuilder &Bldr, ExplodedNode *Pred) {
      const ProgramPoint &L = ProgramPoint::getProgramPoint(S, ProgarmPointKind,
                                Pred->getLocationContext(), checkFn.Checker);
      CheckerContext C(Bldr, Eng, Pred, L);

      // Note, do not pass the statement to the checkers without letting them
      // differentiate if we ran remove dead bindings before or after the
      // statement.
      checkFn(SR, C);
    }
  };
}

/// \brief Run checkers for dead symbols.
void CheckerManager::runCheckersForDeadSymbols(ExplodedNodeSet &Dst,
                                               const ExplodedNodeSet &Src,
                                               SymbolReaper &SymReaper,
                                               const Stmt *S,
                                               ExprEngine &Eng,
                                               ProgramPoint::Kind K) {
  CheckDeadSymbolsContext C(DeadSymbolsCheckers, SymReaper, S, Eng, K);
  expandGraphWithCheckers(C, Dst, Src);
}

/// \brief Run checkers for region changes.
ProgramStateRef
CheckerManager::runCheckersForRegionChanges(ProgramStateRef state,
                                            const InvalidatedSymbols *invalidated,
                                            ArrayRef<const MemRegion *> ExplicitRegions,
                                            ArrayRef<const MemRegion *> Regions,
                                            const LocationContext *LCtx,
                                            const CallEvent *Call) {
  for (unsigned i = 0, e = RegionChangesCheckers.size(); i != e; ++i) {
    // If any checker declares the state infeasible (or if it starts that way),
    // bail out.
    if (!state)
      return nullptr;
    state = RegionChangesCheckers[i](state, invalidated,
                                     ExplicitRegions, Regions,
                                     LCtx, Call);
  }
  return state;
}

/// \brief Run checkers to process symbol escape event.
ProgramStateRef
CheckerManager::runCheckersForPointerEscape(ProgramStateRef State,
                                   const InvalidatedSymbols &Escaped,
                                   const CallEvent *Call,
                                   PointerEscapeKind Kind,
                                   RegionAndSymbolInvalidationTraits *ETraits) {
  assert((Call != nullptr ||
          (Kind != PSK_DirectEscapeOnCall &&
           Kind != PSK_IndirectEscapeOnCall)) &&
         "Call must not be NULL when escaping on call");
    for (unsigned i = 0, e = PointerEscapeCheckers.size(); i != e; ++i) {
      // If any checker declares the state infeasible (or if it starts that
      //  way), bail out.
      if (!State)
        return nullptr;
      State = PointerEscapeCheckers[i](State, Escaped, Call, Kind, ETraits);
    }
  return State;
}

/// \brief Run checkers for handling assumptions on symbolic values.
ProgramStateRef
CheckerManager::runCheckersForEvalAssume(ProgramStateRef state,
                                         SVal Cond, bool Assumption) {
  for (unsigned i = 0, e = EvalAssumeCheckers.size(); i != e; ++i) {
    // If any checker declares the state infeasible (or if it starts that way),
    // bail out.
    if (!state)
      return nullptr;
    state = EvalAssumeCheckers[i](state, Cond, Assumption);
  }
  return state;
}

/// \brief Run checkers for evaluating a call.
/// Only one checker will evaluate the call.
void CheckerManager::runCheckersForEvalCall(ExplodedNodeSet &Dst,
                                            const ExplodedNodeSet &Src,
                                            const CallEvent &Call,
                                            ExprEngine &Eng) {
  const CallExpr *CE = cast<CallExpr>(Call.getOriginExpr());
  for (ExplodedNodeSet::iterator
         NI = Src.begin(), NE = Src.end(); NI != NE; ++NI) {
    ExplodedNode *Pred = *NI;
    bool anyEvaluated = false;

    ExplodedNodeSet checkDst;
    NodeBuilder B(Pred, checkDst, Eng.getBuilderContext());

    // Check if any of the EvalCall callbacks can evaluate the call.
    for (std::vector<EvalCallFunc>::iterator
           EI = EvalCallCheckers.begin(), EE = EvalCallCheckers.end();
         EI != EE; ++EI) {
      ProgramPoint::Kind K = ProgramPoint::PostStmtKind;
      const ProgramPoint &L = ProgramPoint::getProgramPoint(CE, K,
                                Pred->getLocationContext(), EI->Checker);
      bool evaluated = false;
      { // CheckerContext generates transitions(populates checkDest) on
        // destruction, so introduce the scope to make sure it gets properly
        // populated.
        CheckerContext C(B, Eng, Pred, L);
        evaluated = (*EI)(CE, C);
      }
      assert(!(evaluated && anyEvaluated)
             && "There are more than one checkers evaluating the call");
      if (evaluated) {
        anyEvaluated = true;
        Dst.insert(checkDst);
#ifdef NDEBUG
        break; // on release don't check that no other checker also evals.
#endif
      }
    }

    // If none of the checkers evaluated the call, ask ExprEngine to handle it.
    if (!anyEvaluated) {
      NodeBuilder B(Pred, Dst, Eng.getBuilderContext());
      Eng.defaultEvalCall(B, Pred, Call);
    }
  }
}

/// \brief Run checkers for the entire Translation Unit.
void CheckerManager::runCheckersOnEndOfTranslationUnit(
                                                  const TranslationUnitDecl *TU,
                                                  AnalysisManager &mgr,
                                                  BugReporter &BR) {
  for (unsigned i = 0, e = EndOfTranslationUnitCheckers.size(); i != e; ++i)
    EndOfTranslationUnitCheckers[i](TU, mgr, BR);
}

void CheckerManager::runCheckersForPrintState(raw_ostream &Out,
                                              ProgramStateRef State,
                                              const char *NL, const char *Sep) {
  for (llvm::DenseMap<CheckerTag, CheckerRef>::iterator
        I = CheckerTags.begin(), E = CheckerTags.end(); I != E; ++I)
    I->second->printState(Out, State, NL, Sep);
}

//===----------------------------------------------------------------------===//
// Internal registration functions for AST traversing.
//===----------------------------------------------------------------------===//

void CheckerManager::_registerForDecl(CheckDeclFunc checkfn,
                                      HandlesDeclFunc isForDeclFn) {
  DeclCheckerInfo info = { checkfn, isForDeclFn };
  DeclCheckers.push_back(info);
}

void CheckerManager::_registerForBody(CheckDeclFunc checkfn) {
  BodyCheckers.push_back(checkfn);
}

//===----------------------------------------------------------------------===//
// Internal registration functions for path-sensitive checking.
//===----------------------------------------------------------------------===//

void CheckerManager::_registerForPreStmt(CheckStmtFunc checkfn,
                                         HandlesStmtFunc isForStmtFn) {
  StmtCheckerInfo info = { checkfn, isForStmtFn, /*IsPreVisit*/true };
  StmtCheckers.push_back(info);
}
void CheckerManager::_registerForPostStmt(CheckStmtFunc checkfn,
                                          HandlesStmtFunc isForStmtFn) {
  StmtCheckerInfo info = { checkfn, isForStmtFn, /*IsPreVisit*/false };
  StmtCheckers.push_back(info);
}

void CheckerManager::_registerForPreObjCMessage(CheckObjCMessageFunc checkfn) {
  PreObjCMessageCheckers.push_back(checkfn);
}

void CheckerManager::_registerForObjCMessageNil(CheckObjCMessageFunc checkfn) {
  ObjCMessageNilCheckers.push_back(checkfn);
}

void CheckerManager::_registerForPostObjCMessage(CheckObjCMessageFunc checkfn) {
  PostObjCMessageCheckers.push_back(checkfn);
}

void CheckerManager::_registerForPreCall(CheckCallFunc checkfn) {
  PreCallCheckers.push_back(checkfn);
}
void CheckerManager::_registerForPostCall(CheckCallFunc checkfn) {
  PostCallCheckers.push_back(checkfn);
}

void CheckerManager::_registerForLocation(CheckLocationFunc checkfn) {
  LocationCheckers.push_back(checkfn);
}

void CheckerManager::_registerForBind(CheckBindFunc checkfn) {
  BindCheckers.push_back(checkfn);
}

void CheckerManager::_registerForEndAnalysis(CheckEndAnalysisFunc checkfn) {
  EndAnalysisCheckers.push_back(checkfn);
}

void CheckerManager::_registerForBeginFunction(CheckBeginFunctionFunc checkfn) {
  BeginFunctionCheckers.push_back(checkfn);
}

void CheckerManager::_registerForEndFunction(CheckEndFunctionFunc checkfn) {
  EndFunctionCheckers.push_back(checkfn);
}

void CheckerManager::_registerForBranchCondition(
                                             CheckBranchConditionFunc checkfn) {
  BranchConditionCheckers.push_back(checkfn);
}

void CheckerManager::_registerForNewAllocator(CheckNewAllocatorFunc checkfn) {
  NewAllocatorCheckers.push_back(checkfn);
}

void CheckerManager::_registerForLiveSymbols(CheckLiveSymbolsFunc checkfn) {
  LiveSymbolsCheckers.push_back(checkfn);
}

void CheckerManager::_registerForDeadSymbols(CheckDeadSymbolsFunc checkfn) {
  DeadSymbolsCheckers.push_back(checkfn);
}

void CheckerManager::_registerForRegionChanges(CheckRegionChangesFunc checkfn) {
  RegionChangesCheckers.push_back(checkfn);
}

void CheckerManager::_registerForPointerEscape(CheckPointerEscapeFunc checkfn){
  PointerEscapeCheckers.push_back(checkfn);
}

void CheckerManager::_registerForConstPointerEscape(
                                          CheckPointerEscapeFunc checkfn) {
  PointerEscapeCheckers.push_back(checkfn);
}

void CheckerManager::_registerForEvalAssume(EvalAssumeFunc checkfn) {
  EvalAssumeCheckers.push_back(checkfn);
}

void CheckerManager::_registerForEvalCall(EvalCallFunc checkfn) {
  EvalCallCheckers.push_back(checkfn);
}

void CheckerManager::_registerForEndOfTranslationUnit(
                                            CheckEndOfTranslationUnit checkfn) {
  EndOfTranslationUnitCheckers.push_back(checkfn);
}

//===----------------------------------------------------------------------===//
// Implementation details.
//===----------------------------------------------------------------------===//

const CheckerManager::CachedStmtCheckers &
CheckerManager::getCachedStmtCheckersFor(const Stmt *S, bool isPreVisit) {
  assert(S);

  unsigned Key = (S->getStmtClass() << 1) | unsigned(isPreVisit);
  CachedStmtCheckersMapTy::iterator CCI = CachedStmtCheckersMap.find(Key);
  if (CCI != CachedStmtCheckersMap.end())
    return CCI->second;

  // Find the checkers that should run for this Stmt and cache them.
  CachedStmtCheckers &Checkers = CachedStmtCheckersMap[Key];
  for (unsigned i = 0, e = StmtCheckers.size(); i != e; ++i) {
    StmtCheckerInfo &Info = StmtCheckers[i];
    if (Info.IsPreVisit == isPreVisit && Info.IsForStmtFn(S))
      Checkers.push_back(Info.CheckFn);
  }
  return Checkers;
}

CheckerManager::~CheckerManager() {
  for (unsigned i = 0, e = CheckerDtors.size(); i != e; ++i)
    CheckerDtors[i]();
}
