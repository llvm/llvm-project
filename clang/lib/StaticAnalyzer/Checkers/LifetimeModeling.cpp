#include "clang/StaticAnalyzer/Checkers/LifetimeModeling.h"
#include "clang/AST/Attr.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

REGISTER_SET_FACTORY_WITH_PROGRAMSTATE(LifetimeSourceSet, const MemRegion *)
REGISTER_MAP_WITH_PROGRAMSTATE(LifetimeBoundMap, SVal, LifetimeSourceSet)

REGISTER_SET_WITH_PROGRAMSTATE(DeallocatedSourceSet, const MemRegion *)

namespace {

class LifetimeModeling : public Checker<check::PostCall, check::LifetimeEnd, check::DeadSymbols> {
public:
  void printState(raw_ostream &Out, ProgramStateRef State, const char *NL,
                  const char *Sep) const override;
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkLifetimeEnd(const VarDecl *VD, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
};

} // namespace

bool lifetimemodeling::isDeallocated(ProgramStateRef State,
                                     const MemRegion *Region) {
  return State->contains<DeallocatedSourceSet>(Region);
}

static bool getDanglingStackFrame(const MemRegion *Source, ProgramStateRef State, CheckerContext &C) {
  // FIXME: The checker currently handles stack-region sources. Other
  // region kinds require separate methodology. For example, heap
  // regions do not go out of scope at the end of a stack frame, so
  // in order to detect those type of dangling sources the function
  // needs to be expanded to an event-driven approach as well.
  if (const auto *StackSpace = Source->getMemorySpaceAs<StackSpaceRegion>(State)) {
    const StackFrame *SF = StackSpace->getStackFrame();
    const StackFrame *CurrentSF = C.getStackFrame();
    if (SF == CurrentSF || !SF->isParentOf(CurrentSF))
      return true;
  }
  return false;
}

const std::vector<const MemRegion *> lifetimemodeling::checkReturnedBorrower(SVal Val, ProgramStateRef State,
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

void LifetimeModeling::checkLifetimeEnd(const VarDecl *VD,
                                        CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  if (!VD)
    return;

  SVal SourceVal = State->getLValue(VD, C.getStackFrame());
  if (const MemRegion *SourceValRegion = SourceVal.getAsRegion()) {
    State = State->add<DeallocatedSourceSet>(SourceValRegion);
    C.addTransition(State);
  }
}

void LifetimeModeling::checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const {
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

  const auto Sources = State->get<DeallocatedSourceSet>();
  for (const auto *Source : Sources) {
    if (!SymReaper.isLiveRegion(Source))
      State = State->remove<DeallocatedSourceSet>(Source);
  }
  C.addTransition(State);
}

void lifetimemodeling::dumpLifetimeSources(ProgramStateRef State, SVal Source, raw_ostream &OS) {
  const auto *SourceSet = State->get<LifetimeBoundMap>(Source);
  if (!SourceSet)
    return;

  llvm::SmallVector<std::string> RegionNames = to_vector(map_range(llvm::make_pointee_range(*SourceSet), std::mem_fn(&MemRegion::getString)));
  llvm::sort(RegionNames);

  OS << " Origin " << Source << " bound to ";
  llvm::interleaveComma(RegionNames, OS);
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

void ento::registerLifetimeModeling(CheckerManager &Mgr) {
  Mgr.registerChecker<LifetimeModeling>();
}

bool ento::shouldRegisterLifetimeModeling(const CheckerManager &Mgr) {
  return true;
}
