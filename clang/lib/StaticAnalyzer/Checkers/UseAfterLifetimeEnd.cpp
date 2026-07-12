#include "LifetimeModeling.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"

using namespace clang;
using namespace ento;

namespace {
class UseAfterLifetimeEnd : public Checker<check::EndFunction> {
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

  std::vector<const MemRegion *> RetValRegion =
      lifetime_modeling::getDanglingRegionsAfterReturn(RetVal, State, C);
  if (RetValRegion.empty())
    return;

  if (ExplodedNode *N =
          C.generateNonFatalErrorNode(State, C.getPredecessor())) {
    for (const MemRegion *R : RetValRegion)
      reportDanglingSource(R, N, C);
  }
}

void UseAfterLifetimeEnd::reportDanglingSource(const MemRegion *Source,
                                               ExplodedNode *N,
                                               CheckerContext &C) const {
  auto BR = std::make_unique<PathSensitiveBugReport>(
      BugMsg,
      (llvm::Twine("Returning value bound to '") + Source->getString() +
       "' that will go out of scope"),
      N);
  C.emitReport(std::move(BR));
}

void ento::registerUseAfterLifetimeEnd(CheckerManager &Mgr) {
  Mgr.registerChecker<UseAfterLifetimeEnd>();
}

bool ento::shouldRegisterUseAfterLifetimeEnd(const CheckerManager &Mgr) {
  return true;
}
