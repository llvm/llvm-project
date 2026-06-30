#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Checkers/LifetimeModeling.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

namespace {
class UseAfterLifetimeEnd
    : public Checker<check::EndFunction> {
public:
  void reportDanglingSource(const MemRegion *Source, ExplodedNode *N,
                            CheckerContext &C) const;
  void checkEndFunction(const ReturnStmt *RS, CheckerContext &C) const;
  const BugType BugMsg{this, "UseAfterLifetimeEnd", "LifetimeBound"};
};

} // namespace

void UseAfterLifetimeEnd::checkEndFunction(const ReturnStmt *RS,
                                           CheckerContext &C) const {
  if (!RS)
    return;

  ProgramStateRef State = C.getState();

  const Expr *RetExpr = RS->getRetValue();
  if (!RetExpr)
    return;

  RetExpr = RetExpr->IgnoreParens();
  SVal RetVal = C.getSVal(RetExpr);
  ExplodedNode *N = nullptr;

  std::vector<const MemRegion *> RetValRegion = lifetimemodeling::checkReturnedBorrower(RetVal, State, C);
  for (const MemRegion *Region : RetValRegion) {
    if (!N)
      N = C.generateNonFatalErrorNode();
    if (!N)
      return;

    reportDanglingSource(Region, N, C);
  }
}

void UseAfterLifetimeEnd::reportDanglingSource(const MemRegion *Source,
                                               ExplodedNode *N,
                                               CheckerContext &C) const {
  auto BR = std::make_unique<PathSensitiveBugReport>(BugMsg, (llvm::Twine("Returning value bound to '") + Source->getString() + "' that will go out of scope"), N);
  C.emitReport(std::move(BR));
}

namespace {
class DebugUseAfterLifetimeEnd : public Checker<eval::Call> {
public:
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void analyzerDumpLifetimeOriginsOf(const CallEvent &Call,
                                     CheckerContext &C) const;

  const BugType BugMsg{this, "DebugUseAfterLifetimeEnd",
                       "DebugUseAfterLifetimeEnd"};
  using FnCheck = void (DebugUseAfterLifetimeEnd::*)(const CallEvent &Call,
                                                     CheckerContext &C) const;

  const CallDescriptionMap<FnCheck> Callbacks = {
      {{CDM::SimpleFunc, {"clang_analyzer_dumpLifetimeOriginsOf"}},
       &DebugUseAfterLifetimeEnd::analyzerDumpLifetimeOriginsOf},
  };
};

} // namespace

bool DebugUseAfterLifetimeEnd::evalCall(const CallEvent &Call,
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

void DebugUseAfterLifetimeEnd::analyzerDumpLifetimeOriginsOf(
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
  llvm::SmallString<128> Str;
  llvm::raw_svector_ostream OS(Str);
  lifetimemodeling::dumpLifetimeSources(State, ArgSVal, OS);

  if (!Str.empty()) {
    if (ExplodedNode *N = C.generateNonFatalErrorNode())
      C.emitReport(std::make_unique<PathSensitiveBugReport>(BugMsg, OS.str(), N));
  }
}

void ento::registerUseAfterLifetimeEnd(CheckerManager &Mgr) {
  Mgr.registerChecker<UseAfterLifetimeEnd>();
}

bool ento::shouldRegisterUseAfterLifetimeEnd(const CheckerManager &Mgr) {
  return true;
}

void ento::registerDebugUseAfterLifetimeEnd(CheckerManager &Mgr) {
  Mgr.registerChecker<DebugUseAfterLifetimeEnd>();
}

bool ento::shouldRegisterDebugUseAfterLifetimeEnd(const CheckerManager &Mgr) {
  return true;
}
