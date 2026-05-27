#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include <AllocationState.h>

using namespace clang;
using namespace ento;

REGISTER_MAP_WITH_PROGRAMSTATE(LifetimeBoundMap, const MemRegion *,
                               const MemRegion *);

class LifetimeAnnotations : public Checker<check::PostCall> {
public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void printState(raw_ostream &Out, ProgramStateRef State, const char *NL,
                  const char *Sep) const override;
};

void LifetimeAnnotations::checkPostCall(const CallEvent &Call,
                                        CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  const auto *MethodDecl = dyn_cast_if_present<CXXMethodDecl>(Call.getDecl());

  if (!MethodDecl)
    return;

  unsigned LBParamIdx = MethodDecl->getNumParams();
  for (unsigned i = 0; i < MethodDecl->getNumParams(); i++) {
    if (MethodDecl->getParamDecl(i)->hasAttr<LifetimeBoundAttr>()) {
      LBParamIdx = i;
      break;
    }
  }
  if (LBParamIdx == MethodDecl->getNumParams())
    return;

  SVal RetVal = Call.getReturnValue();
  const MemRegion *RetValRegion = RetVal.getAsRegion();
  if (!RetValRegion)
    return;

  SVal ArgVal = Call.getArgSVal(LBParamIdx);
  const MemRegion *ArgValRegion = ArgVal.getAsRegion();
  if (!ArgValRegion)
    return;

  State = State->set<LifetimeBoundMap>(RetValRegion, ArgValRegion);
  C.addTransition(State);
}

void LifetimeAnnotations::printState(raw_ostream &Out, ProgramStateRef State,
                                     const char *NL, const char *Sep) const {
  auto LBTy = State->get<LifetimeBoundMap>();

  if (!LBTy.isEmpty()) {
    Out << Sep << "LifetimeBound objects: ";

    for (auto I : LBTy) {
      Out << I.first << " bound to " << I.second << NL;
    }
  }
}

void ento::registerLifetimeAnnotations(CheckerManager &mgr) {
  mgr.registerChecker<LifetimeAnnotations>();
}

bool ento::shouldRegisterLifetimeAnnotations(const CheckerManager &mgr) {
  return true;
}
