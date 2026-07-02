#include "clang/StaticAnalyzer/Checkers/LifetimeModeling.h"
#include "clang/AST/Attr.h"
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
REGISTER_MAP_WITH_PROGRAMSTATE(LifetimeBoundMap, SVal, LifetimeSourceSet)

namespace {

class LifetimeModeling : public Checker<check::PostCall, check::DeadSymbols> {
public:
  void printState(raw_ostream &Out, ProgramStateRef State, const char *NL,
                  const char *Sep) const override;
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
};

} // namespace

static bool getDanglingStackFrame(const MemRegion *Source,
                                  ProgramStateRef State, CheckerContext &C) {
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

const std::vector<const MemRegion *>
lifetime_modeling::checkReturnedBorrower(SVal Val, ProgramStateRef State,
                                         CheckerContext &C) {
  std::vector<const MemRegion *> Regions;
  if (auto *SourceSet = State->get<LifetimeBoundMap>(Val)) {
    for (const MemRegion *Region : *SourceSet) {
      if (getDanglingStackFrame(Region, State, C))
        Regions.push_back(Region);
    }
  }
  return Regions;
}

static ProgramStateRef bindValues(ProgramStateRef State, SVal RetVal,
                                  const MemRegion *Source) {
  LifetimeSourceSet::Factory &F = State->get_context<LifetimeSourceSet>();
  const LifetimeSourceSet *LSet = State->get<LifetimeBoundMap>(RetVal);

  LifetimeSourceSet Set = LSet ? *LSet : F.getEmptySet();
  Set = F.add(Set, Source);
  State = State->set<LifetimeBoundMap>(RetVal, Set);
  return State;
}

void LifetimeModeling::checkPostCall(const CallEvent &Call,
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
      if (const MemRegion *AttrRegion = IC->getCXXThisVal().getAsRegion())
        State = bindValues(State, RetVal, AttrRegion);
    }
  }
  C.addTransition(State);
}

void LifetimeModeling::checkDeadSymbols(SymbolReaper &SymReaper,
                                        CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  LifetimeBoundMapTy LBMap = State->get<LifetimeBoundMap>();

  for (SVal Val : llvm::make_first_range(LBMap)) {
    if (const MemRegion *ValRegion = Val.getAsRegion()) {
      if (!SymReaper.isLiveRegion(ValRegion))
        State = State->remove<LifetimeBoundMap>(Val);
    } else if (SymbolRef ValRef =
                   Val.getAsSymbol(/*IncludeBaseRegions=*/true)) {
      if (!SymReaper.isLive(ValRef))
        State = State->remove<LifetimeBoundMap>(Val);
    }
  }
  C.addTransition(State);
}

void LifetimeModeling::printState(raw_ostream &Out, ProgramStateRef State,
                                  const char *NL, const char *Sep) const {
  auto LBMap = State->get<LifetimeBoundMap>();

  if (LBMap.isEmpty())
    return;

  Out << Sep << "LifetimeBound bindings:" << NL;
  for (auto &&[OriginSym, SourceSet] : LBMap) {
    for (const auto *Region : SourceSet)
      Out << " Origin " << OriginSym << " contains Loan " << Region << NL;
  }
}

namespace {
class DebugLifetimeModeling : public Checker<eval::Call> {
public:
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void analyzerDumpLifetimeOriginsOf(const CallEvent &Call,
                                     CheckerContext &C) const;
  const BugType BugMsg{this, "DebugLifetimeModeling", "DebugLifetimeModeling"};
  using FnCheck = void (DebugLifetimeModeling::*)(const CallEvent &Call,
                                                  CheckerContext &C) const;

  const CallDescriptionMap<FnCheck> Callbacks = {
      {{CDM::SimpleFunc, {"clang_analyzer_dumpLifetimeOriginsOf"}},
       &DebugLifetimeModeling::analyzerDumpLifetimeOriginsOf},
  };
};

} // namespace

bool DebugLifetimeModeling::evalCall(const CallEvent &Call,
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

void DebugLifetimeModeling::analyzerDumpLifetimeOriginsOf(
    const CallEvent &Call, CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  if (Call.getNumArgs() != 1) {
    if (ExplodedNode *N = C.generateNonFatalErrorNode()) {
      auto BR = std::make_unique<PathSensitiveBugReport>(
          BugMsg,
          "clang_analyzer_dumpLifetimeOriginsOf requires exactly 1 argument",
          N);
      C.emitReport(std::move(BR));
    }
    return;
  }

  SVal ArgSVal = Call.getArgSVal(0);
  const LifetimeSourceSet *SourceSet = State->get<LifetimeBoundMap>(ArgSVal);

  if (!SourceSet)
    return;

  if (ExplodedNode *N = C.generateNonFatalErrorNode()) {
    llvm::SmallVector<std::string> RegionNames =
        to_vector(map_range(llvm::make_pointee_range(*SourceSet),
                            std::mem_fn(&MemRegion::getString)));
    llvm::sort(RegionNames);

    llvm::SmallString<128> Str;
    llvm::raw_svector_ostream OS(Str);
    OS << " Origin " << ArgSVal << " bound to ";
    llvm::interleaveComma(RegionNames, OS);
    C.emitReport(std::make_unique<PathSensitiveBugReport>(BugMsg, OS.str(), N));
  }
}

void ento::registerLifetimeModeling(CheckerManager &Mgr) {
  Mgr.registerChecker<LifetimeModeling>();
}

bool ento::shouldRegisterLifetimeModeling(const CheckerManager &Mgr) {
  return true;
}

void ento::registerDebugLifetimeModeling(CheckerManager &Mgr) {
  Mgr.registerChecker<DebugLifetimeModeling>();
}

bool ento::shouldRegisterDebugLifetimeModeling(const CheckerManager &Mgr) {
  return true;
}
