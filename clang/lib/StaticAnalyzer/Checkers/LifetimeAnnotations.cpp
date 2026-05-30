#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "AllocationState.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"


using namespace clang;
using namespace ento;
using namespace clang::lifetimes;

REGISTER_MAP_WITH_PROGRAMSTATE(LifetimeBoundMap, SymbolRef,
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

  const auto *FC = dyn_cast_if_present<AnyFunctionCall>(&Call);
  if (!FC)
    return;

  const FunctionDecl *FD = FC->getDecl();
  if (!FD)
    return;

  unsigned LBParamIdx = FD->getNumParams();
  // FIXME: Use range based for loop instead. Currently that would require
  // to also change how we create ArgVal which would need a new logic to
  // be implemented.
  for (unsigned I = 0, E = FD->getNumParams(); I != E; I++) {
    if (FD->getParamDecl(I)->hasAttr<LifetimeBoundAttr>()) {
      LBParamIdx = I;
      // FIXME: If multiple parameters are annotated this logic would
      // prevent the analyzer to read after the first parameter.
      break;
    }
  }
  SVal RetVal = Call.getReturnValue();

  SymbolRef RetValSym = RetVal.getAsSymbol(/*IncludeBaseRegions=*/true);
  if(!RetValSym)
    return;

  if (LBParamIdx != FD->getNumParams()) {
    SVal ArgVal = Call.getArgSVal(LBParamIdx);
    const MemRegion *ArgValRegion = ArgVal.getAsRegion();
    // FIXME: if(!ArgValRegion) should be also handled since in those cases
    // the argument has no region, but still needs to be tracked.
    if (ArgValRegion)
        State = State->set<LifetimeBoundMap>(RetValSym, ArgValRegion);
  }

  if (const auto *IC = dyn_cast<CXXInstanceCall>(&Call)) {
    if (implicitObjectParamIsLifetimeBound(FD)) {
      const MemRegion *AttrRegion = IC->getCXXThisVal().getAsRegion();

      if (AttrRegion)
          State = State->set<LifetimeBoundMap>(RetValSym, AttrRegion);
    }
  }
  C.addTransition(State);
}

void LifetimeAnnotations::printState(raw_ostream &Out, ProgramStateRef State,
                                     const char *NL, const char *Sep) const {
  auto LBVal = State->get<LifetimeBoundMap>();

  if (LBVal.isEmpty())
    return;

  Out << Sep << "LifetimeBound bindings:" << NL;
  for (auto&& [RetValSym, ArgValRegion] : LBVal) {
    Out << " Origin " << RetValSym << " contains Loan " << ArgValRegion << NL;
  }
}

void ento::registerLifetimeAnnotations(CheckerManager &mgr) {
  mgr.registerChecker<LifetimeAnnotations>();
}

bool ento::shouldRegisterLifetimeAnnotations(const CheckerManager &mgr) {
  return true;
}
