#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

REGISTER_SET_FACTORY_WITH_PROGRAMSTATE(LifetimeSourceSet, const MemRegion *)
REGISTER_MAP_WITH_PROGRAMSTATE(LifetimeBoundMap, SymbolRef, LifetimeSourceSet)

REGISTER_MAP_WITH_PROGRAMSTATE(LifetimeBoundMapVal, const MemRegion *,
                               LifetimeSourceSet)
REGISTER_SET_WITH_PROGRAMSTATE(DeallocatedSourceSet, const MemRegion *)

namespace {
class LifetimeAnnotations
    : public Checker<check::PostCall, check::EndFunction, check::Location> {
public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void printState(raw_ostream &Out, ProgramStateRef State, const char *NL,
                  const char *Sep) const override;
  ProgramStateRef bindValues(ProgramStateRef State, SVal RetVal,
                             const MemRegion *Source) const;
  bool hasDanglingSource(const MemRegion *Source, ProgramStateRef State,
                         CheckerContext &C) const;
  void reportDanglingSource(const MemRegion *Region, ExplodedNode *N,
                            CheckerContext &C) const;
  void reportUseAfterScope(const MemRegion *Region, ExplodedNode *N,
                           CheckerContext &C) const;

  template <typename MapTy, typename KeyTy>
  void checkReturnedBorrower(const MapTy &Map, const KeyTy RetKey,
                             ProgramStateRef State, CheckerContext &C) const;
  void reportDanglingBorrower(const LifetimeSourceSet *Sources,
                              CheckerContext &C) const;
  void checkEndFunction(const ReturnStmt *RS, CheckerContext &C) const;
  void checkLocation(SVal Loc, bool IsLoad, const Stmt *S,
                     CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;

  const BugType BugMsg{this, "LifetimeAnnotations", "LifetimeBound"};
};

} // namespace

ProgramStateRef LifetimeAnnotations::bindValues(ProgramStateRef State,
                                                SVal RetVal,
                                                const MemRegion *Source) const {
  LifetimeSourceSet::Factory &F =
      State->getStateManager().get_context<LifetimeSourceSet>();

  if (SymbolRef RetValSym = RetVal.getAsSymbol(/*IncludeBaseRegions=*/true)) {
    const LifetimeSourceSet *LBSet = State->get<LifetimeBoundMap>(RetValSym);
    LifetimeSourceSet Set = LBSet ? *LBSet : F.getEmptySet();
    Set = F.add(Set, Source);
    State = State->set<LifetimeBoundMap>(RetValSym, Set);
  } else if (const MemRegion *RetValRegion = RetVal.getAsRegion()) {
    const LifetimeSourceSet *LBValSet =
        State->get<LifetimeBoundMapVal>(RetValRegion);
    LifetimeSourceSet Set = LBValSet ? *LBValSet : F.getEmptySet();
    Set = F.add(Set, Source);
    State = State->set<LifetimeBoundMapVal>(RetValRegion, Set);
  }
  return State;
}

void LifetimeAnnotations::checkPostCall(const CallEvent &Call,
                                        CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  const auto *FC = dyn_cast<AnyFunctionCall>(&Call);
  if (!FC)
    return;

  const FunctionDecl *FD = FC->getDecl();
  if (!FD)
    return;

  SVal RetVal = Call.getReturnValue();

  for (const ParmVarDecl *PVD : FD->parameters()) {
    if (PVD->hasAttr<LifetimeBoundAttr>()) {
      unsigned Idx = PVD->getFunctionScopeIndex();
      SVal Arg = Call.getArgSVal(Idx);
      if (const MemRegion *ArgValRegion = Arg.getAsRegion())
        State = bindValues(State, RetVal, ArgValRegion);
    }
  }

  if (const auto *IC = dyn_cast<CXXInstanceCall>(&Call)) {
    if (lifetimes::implicitObjectParamIsLifetimeBound(FD)) {
      if (const MemRegion *AttrRegion = IC->getCXXThisVal().getAsRegion()) {
        State = bindValues(State, RetVal, AttrRegion);
      }
    }
  }
  C.addTransition(State);
}

bool LifetimeAnnotations::hasDanglingSource(const MemRegion *Source,
                                            ProgramStateRef State,
                                            CheckerContext &C) const {
  // FIXME: The checker currently handles stack-region sources. Other
  // region kinds require separate methodology. For example, heap
  // regions do not go out of scope at the end of a stack frame, so
  // in order to detect those type of dangling sources the function
  // needs to be expanded to an event-driven approach as well.
  if (const auto *StackSpace =
          Source->getMemorySpaceAs<StackSpaceRegion>(State)) {
    const StackFrame *SF = StackSpace->getStackFrame();
    const StackFrame *CurrentSF = C.getStackFrame();
    if (SF == CurrentSF || !SF->isParentOf(CurrentSF))
      return true;
  }
  return false;
}

template <typename MapTy, typename KeyTy>
void LifetimeAnnotations::checkReturnedBorrower(const MapTy &Map,
                                                const KeyTy RetKey,
                                                ProgramStateRef State,
                                                CheckerContext &C) const {
  for (auto &&[Origin, SourceSet] : Map) {
    if (Origin == RetKey) {
      for (const MemRegion *Region : SourceSet) {
        if (hasDanglingSource(Region, State, C))
          if (ExplodedNode *N = C.generateNonFatalErrorNode())
            reportDanglingSource(Region, N, C);
      }
    }
  }
}

void LifetimeAnnotations::checkEndFunction(const ReturnStmt *RS,
                                           CheckerContext &C) const {
  if (!RS)
    return;

  ProgramStateRef State = C.getState();
  auto LBMap = State->get<LifetimeBoundMap>();
  auto LBMapVal = State->get<LifetimeBoundMapVal>();

  if (LBMap.isEmpty() && LBMapVal.isEmpty())
    return;

  const Expr *RetExpr = RS->getRetValue();
  if (!RetExpr)
    return;

  RetExpr = RetExpr->IgnoreParens();
  SVal RetVal = C.getSVal(RetExpr);

  SymbolRef RetValSym = RetVal.getAsSymbol(/*IncludeBaseRegions=*/true);
  const MemRegion *RetValRegion = RetVal.getAsRegion();
  if (!RetValSym && !RetValRegion)
    return;

  if (RetValSym)
    checkReturnedBorrower(LBMap, RetValSym, State, C);

  if (RetValRegion)
    checkReturnedBorrower(LBMapVal, RetValRegion, State, C);
}

void LifetimeAnnotations::reportDanglingBorrower(
    const LifetimeSourceSet *Sources, CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  for (const MemRegion *Source : *Sources) {
    if (State->contains<DeallocatedSourceSet>(Source)) {
      if (ExplodedNode *N = C.generateNonFatalErrorNode())
        reportUseAfterScope(Source, N, C);
    }
  }
}

void LifetimeAnnotations::checkLocation(SVal Loc, bool IsLoad, const Stmt *S,
                                        CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  auto LBMap = State->get<LifetimeBoundMap>();
  auto LBMapVal = State->get<LifetimeBoundMapVal>();

  if (LBMap.isEmpty() && LBMapVal.isEmpty())
    return;

  // FIXME: If a borrower has multiple bound sources the callback
  // warns if any of the sources have died. PathDiagnosticVisitor
  // should be used to trace and identify which annotated parameter
  // recorded the binding. Attaching this information as path notes
  // would make the diagnostics more useful to the user.
  if (SymbolRef LocSym = Loc.getAsSymbol(true)) {
    if (auto *SourceSet = State->get<LifetimeBoundMap>(LocSym))
      reportDanglingBorrower(SourceSet, C);
  }

  if (const MemRegion *LocRegion = Loc.getAsRegion()) {
    if (auto *SourceSet = State->get<LifetimeBoundMapVal>(LocRegion))
      reportDanglingBorrower(SourceSet, C);
  }
}

void LifetimeAnnotations::reportDanglingSource(const MemRegion *Region,
                                               ExplodedNode *N,
                                               CheckerContext &C) const {
  std::string ErrorMessage =
      (llvm::Twine("Returning value bound to '") + Region->getString() +
       "' that will go out of scope")
          .str();
  auto BR = std::make_unique<PathSensitiveBugReport>(BugMsg, ErrorMessage, N);
  C.emitReport(std::move(BR));
}

void LifetimeAnnotations::reportUseAfterScope(const MemRegion *Region,
                                              ExplodedNode *N,
                                              CheckerContext &C) const {
  std::string ErrorMessage = (llvm::Twine("Use of '") + Region->getString() +
                              "' after its lifetime ended.")
                                 .str();
  auto BR = std::make_unique<PathSensitiveBugReport>(BugMsg, ErrorMessage, N);
  C.emitReport(std::move(BR));
}

void LifetimeAnnotations::checkDeadSymbols(SymbolReaper &SymReaper,
                                           CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  // Get both maps since both of them needs to be cleaned up
  LifetimeBoundMapTy LBMap = State->get<LifetimeBoundMap>();
  LifetimeBoundMapValTy LBMapVal = State->get<LifetimeBoundMapVal>();

  for (const SymExpr *Sym : llvm::make_first_range(LBMap)) {
    if (!SymReaper.isLive(Sym))
      State = State->remove<LifetimeBoundMap>(Sym);
  }

  for (const MemRegion *Region : llvm::make_first_range(LBMapVal)) {
    if (!SymReaper.isLiveRegion(Region))
      State = State->remove<LifetimeBoundMapVal>(Region);
  }
  if (State != C.getState())
    C.addTransition(State);
}

void LifetimeAnnotations::printState(raw_ostream &Out, ProgramStateRef State,
                                     const char *NL, const char *Sep) const {
  auto LBMap = State->get<LifetimeBoundMap>();
  auto LBMapVal = State->get<LifetimeBoundMapVal>();

  if (LBMap.isEmpty() && LBMapVal.isEmpty())
    return;

  Out << Sep << "LifetimeBound bindings:" << NL;
  for (auto &&[OriginSym, SourceSet] : LBMap) {
    for (const auto *Region : SourceSet)
      Out << " Origin " << OriginSym << " contains Loan " << Region << NL;
  }
  for (auto &&[OriginRegion, SourceSet] : LBMapVal) {
    for (const auto *Region : SourceSet)
      Out << " Origin " << OriginRegion << " contains Loan " << Region << NL;
  }
}

namespace {
class DebugLifetimeAnnotations : public Checker<eval::Call> {
public:
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void analyzerLifetimeBound(const CallEvent &Call, CheckerContext &C) const;

  const BugType BugMsg{this, "DebugLifetimeAnnotations", "DebugLifetimeBound"};
  using FnCheck = void (DebugLifetimeAnnotations::*)(const CallEvent &Call,
                                                     CheckerContext &C) const;

  const CallDescriptionMap<FnCheck> Callbacks = {
      {{CDM::SimpleFunc, {"clang_analyzer_lifetime_bound"}},
       &DebugLifetimeAnnotations::analyzerLifetimeBound},
  };
};

} // namespace

bool DebugLifetimeAnnotations::evalCall(const CallEvent &Call,
                                        CheckerContext &C) const {

  const auto *CE = dyn_cast_if_present<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return false;

  const FnCheck *Handler = Callbacks.lookup(Call);
  if (!Handler)
    return false;

  (this->*(*Handler))(Call, C);
  return true;
}

void DebugLifetimeAnnotations::analyzerLifetimeBound(const CallEvent &Call,
                                                     CheckerContext &C) const {

  ProgramStateRef State = C.getState();
  unsigned int ArgCount = Call.getNumArgs();
  if (ArgCount != 1)
    return;

  SVal ArgSVal = Call.getArgSVal(0);

  const MemRegion *ArgValRegion = ArgSVal.getAsRegion();
  SymbolRef ArgSValSym = ArgSVal.getAsSymbol(/*IncludeBaseRegions=*/true);

  llvm::SmallString<128> Str;
  llvm::raw_svector_ostream OS(Str);
  ExplodedNode *N = C.generateNonFatalErrorNode();
  if (!N)
    return;

  if (ArgSValSym) {
    if (auto *SourceSet = State->get<LifetimeBoundMap>(ArgSValSym)) {
      for (const auto *Region : *SourceSet) {
        OS << " Origin " << ArgSValSym << " bound to " << Region;
        auto BR = std::make_unique<PathSensitiveBugReport>(BugMsg, OS.str(), N);
        C.emitReport(std::move(BR));
        Str.clear();
      }
    }
  }

  if (ArgValRegion) {
    if (auto *SourceSet = State->get<LifetimeBoundMapVal>(ArgValRegion)) {
      for (const auto *Region : *SourceSet) {
        OS << " Origin " << ArgValRegion << " bound to " << Region;
        auto BR = std::make_unique<PathSensitiveBugReport>(BugMsg, OS.str(), N);
        C.emitReport(std::move(BR));
        Str.clear();
      }
    }
  }
}

void ento::registerLifetimeAnnotations(CheckerManager &mgr) {
  mgr.registerChecker<LifetimeAnnotations>();
}

bool ento::shouldRegisterLifetimeAnnotations(const CheckerManager &mgr) {
  return true;
}

void ento::registerDebugLifetimeAnnotations(CheckerManager &mgr) {
  mgr.registerChecker<DebugLifetimeAnnotations>();
}

bool ento::shouldRegisterDebugLifetimeAnnotations(const CheckerManager &mgr) {
  return true;
}
