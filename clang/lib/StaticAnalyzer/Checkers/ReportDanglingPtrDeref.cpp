#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Checkers/LifetimeModeling.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class ReportDanglingPtrDeref : public Checker<check::Location> {
public:
  void checkLocation(SVal Loc, bool IsLoad, const Stmt *S,
                     CheckerContext &C) const;
  void reportUseAfterScope(const MemRegion *Region, ExplodedNode *N,
                           CheckerContext &C) const;
  const BugType BugMsg{this, "ReportDanglingPtrDeref", "LifetimeBound"};
};
} // namespace

void ReportDanglingPtrDeref::checkLocation(SVal Loc, bool IsLoad, const Stmt *S,
                                           CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  if (const MemRegion *LocRegion = Loc.getAsRegion()) {
    if (lifetimemodeling::isDeallocated(State, LocRegion)) {
      if (ExplodedNode *N = C.generateNonFatalErrorNode())
        reportUseAfterScope(LocRegion, N, C);
    }
  }
}

void ReportDanglingPtrDeref::reportUseAfterScope(const MemRegion *Region,
                                                 ExplodedNode *N,
                                                 CheckerContext &C) const {
  auto BR = std::make_unique<PathSensitiveBugReport>(
      BugMsg,
      (llvm::Twine("Use of '") + Region->getString() +
       "' after its lifetime ended.")
          .str(),
      N);
  C.emitReport(std::move(BR));
}

void ento::registerReportDanglingPtrDeref(CheckerManager &Mgr) {
  Mgr.registerChecker<ReportDanglingPtrDeref>();
}

bool ento::shouldRegisterReportDanglingPtrDeref(const CheckerManager &Mgr) {
  return true;
}
