#include "clang/StaticAnalyzer/Checkers/LifetimeModeling.h"
#include "clang/AST/Attr.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

REGISTER_SET_FACTORY_WITH_PROGRAMSTATE(LifetimeSourceSet, const MemRegion *)
REGISTER_MAP_WITH_PROGRAMSTATE(LifetimeBoundMap, SVal, LifetimeSourceSet)

REGISTER_SET_WITH_PROGRAMSTATE(DeallocatedSourceSet, const MemRegion *)

namespace {

class LifetimeModeling : public Checker<check::PostCall, check::LifetimeEnd> {
public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkLifetimeEnd(const VarDecl *VD, CheckerContext &C) const;
};

} // namespace

namespace clang {
namespace ento {
namespace lifetimemodeling {
std::vector<const MemRegion *> getLifetimeSourceSet(ProgramStateRef State,
                                                    SVal Val) {
  std::vector<const MemRegion *> StoreRegion;
  if (const auto *SourceSet = State->get<LifetimeBoundMap>(Val)) {
    for (const MemRegion *Region : *SourceSet)
      StoreRegion.push_back(Region);
  }
  return StoreRegion;
}

bool isDeallocated(ProgramStateRef State, const MemRegion *Region) {
  return State->contains<DeallocatedSourceSet>(Region);
}

} // namespace lifetimemodeling
} // namespace ento
} // namespace clang

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

void ento::registerLifetimeModeling(CheckerManager &Mgr) {
  Mgr.registerChecker<LifetimeModeling>();
}

bool ento::shouldRegisterLifetimeModeling(const CheckerManager &Mgr) {
  return true;
}
