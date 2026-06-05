#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/Support/raw_ostream.h"
#include "AllocationState.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"


using namespace clang;
using namespace ento;
using namespace clang::lifetimes;

REGISTER_MAP_WITH_PROGRAMSTATE(LifetimeBoundMap, SymbolRef,
                               const MemRegion *);
REGISTER_MAP_WITH_PROGRAMSTATE(LifetimeBoundMapVal, const MemRegion *, const MemRegion *);


class LifetimeAnnotations : public Checker<check::PostCall, eval::Call> {
public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void printState(raw_ostream &Out, ProgramStateRef State, const char *NL,
                  const char *Sep) const override;
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void analyzerLifetimeBound(const CallEvent &Call, const CallExpr *, CheckerContext &C) const;

  const BugType BugMsg{this, "LifetimeAnnotations", "LifetimeBound"};
};

typedef void (LifetimeAnnotations::*FnCheck)(const CallEvent &Call, const CallExpr *,
                                              CheckerContext &) const;
CallDescriptionMap<FnCheck> Callbacks = {
  {{CDM::SimpleFunc, {"clang_analyzer_lifetime_bound"}},
    &LifetimeAnnotations::analyzerLifetimeBound},
};

void LifetimeAnnotations::checkPostCall(const CallEvent &Call,
                                        CheckerContext &C) const {
  llvm::errs() << "checkPostCall fired" << "\n";
  ProgramStateRef State = C.getState();

  const auto *FC = dyn_cast_if_present<AnyFunctionCall>(&Call);
  if (!FC)
    return;

  const FunctionDecl *FD = FC->getDecl();
  if (!FD)
    return;

  SVal RetVal = Call.getReturnValue();
  SymbolRef RetValSym = RetVal.getAsSymbol(/*IncludeBaseRegions=*/true);
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

  if (LBParamIdx != FD->getNumParams()) {
    SVal ArgVal = Call.getArgSVal(LBParamIdx);
    if (const MemRegion *ArgValRegion = ArgVal.getAsRegion()) {
      if (RetValSym)
        State = State->set<LifetimeBoundMap>(RetValSym, ArgValRegion);
      if (const MemRegion *RetValRegion = RetVal.getAsRegion())
        State = State->set<LifetimeBoundMapVal>(RetValRegion, ArgValRegion);
    }
  }

  if (const auto *IC = dyn_cast<CXXInstanceCall>(&Call)) {
    if (implicitObjectParamIsLifetimeBound(FD)) {
      if (const MemRegion *AttrRegion = IC->getCXXThisVal().getAsRegion()) {
        if (RetValSym)
          State = State->set<LifetimeBoundMap>(RetValSym, AttrRegion);
        if (const MemRegion *RetValRegion = RetVal.getAsRegion())
          State = State->set<LifetimeBoundMapVal>(RetValRegion, AttrRegion);
      }
    }
  }
  C.addTransition(State);
}

void LifetimeAnnotations::printState(raw_ostream &Out, ProgramStateRef State,
                                     const char *NL, const char *Sep) const {
  auto LBMap = State->get<LifetimeBoundMap>();
  auto LBMapVal = State->get<LifetimeBoundMapVal>();

  if (LBMap.isEmpty() && LBMapVal.isEmpty())
    return;

  Out << Sep << "LifetimeBound bindings:" << NL;
  for (auto&& [RetValSym, ArgValRegion] : LBMap) {
    Out << " Origin " << RetValSym << " contains Loan " << ArgValRegion << NL;
  }
  for (auto&& [RetVal, ArgValRegion]: LBMapVal) {
    Out << " Origin " << RetVal << " contains Loan " << ArgValRegion << NL;
  }
}

bool LifetimeAnnotations::evalCall(const CallEvent &Call, CheckerContext &C) const {

  const auto *CE = llvm::dyn_cast_if_present<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return false;

  const FnCheck *Handler = Callbacks.lookup(Call);
  if (!Handler)
    return false;

  (this->*(*Handler))(Call, CE, C);
  return true;
  C.addTransition(C.getState());
}

void LifetimeAnnotations::analyzerLifetimeBound(const CallEvent &Call, const CallExpr *CE, CheckerContext &C) const {

  ProgramStateRef State = C.getState();
  unsigned int ArgExpr = CE->getNumArgs();
  if (ArgExpr != 1)
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
    if (const auto *ArgValLookFor = State->get<LifetimeBoundMap>(ArgSValSym)) {
      OS << " Origin " << ArgSValSym << " contains loan " << *ArgValLookFor;
      auto BR = std::make_unique<PathSensitiveBugReport>(BugMsg, OS.str(), N);
      C.emitReport(std::move(BR));
      Str.clear();
    }
  }

  if (ArgValRegion) {
    if (const auto *AttrValLookFor = State->get<LifetimeBoundMapVal>(ArgValRegion)) {
      OS << " Origin " << ArgValRegion << " bound to " << *AttrValLookFor;
      auto BR = std::make_unique<PathSensitiveBugReport>(BugMsg, OS.str(), N);
      C.emitReport(std::move(BR));
      Str.clear();
    }
  }
}

void ento::registerLifetimeAnnotations(CheckerManager &mgr) {
  mgr.registerChecker<LifetimeAnnotations>();
}

bool ento::shouldRegisterLifetimeAnnotations(const CheckerManager &mgr) {
  return true;
}
