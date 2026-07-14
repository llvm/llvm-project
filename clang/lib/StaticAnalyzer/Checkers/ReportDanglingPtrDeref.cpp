#include "LifetimeModeling.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class DanglingPtrDeref : public Checker<check::Location> {
public:
  void checkLocation(SVal Loc, bool IsLoad, const Stmt *S,
                     CheckerContext &C) const;
  void reportUseAfterScope(const MemRegion *Region, ExplodedNode *N,
                           CheckerContext &C) const;
  const BugType BugMsg{this, "ReportDanglingPtrDeref", "LifetimeBound"};
};

class DanglingPtrDerefBRVisitor : public BugReporterVisitor {
  const MemRegion *SourceRegion;

public:
  explicit DanglingPtrDerefBRVisitor(const MemRegion *Source)
      : SourceRegion(Source) {}

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    ID.AddPointer(SourceRegion);
  }

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override;
};

} // namespace

void DanglingPtrDeref::checkLocation(SVal Loc, bool IsLoad, const Stmt *S,
                                     CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  if (const MemRegion *LocRegion = Loc.getAsRegion()) {
    if (lifetime_modeling::isDeallocated(State, LocRegion)) {
      if (ExplodedNode *N = C.generateNonFatalErrorNode(State))
        reportUseAfterScope(LocRegion, N, C);
    }
  }
}

void DanglingPtrDeref::reportUseAfterScope(const MemRegion *Region,
                                           ExplodedNode *N,
                                           CheckerContext &C) const {
  auto BR = std::make_unique<PathSensitiveBugReport>(
      BugMsg,
      (llvm::Twine("Use of '") + Region->getString() +
       "' after its lifetime ended."),
      N);
  BR->addVisitor<DanglingPtrDerefBRVisitor>(Region);
  C.emitReport(std::move(BR));
}

PathDiagnosticPieceRef
DanglingPtrDerefBRVisitor::VisitNode(const ExplodedNode *N,
                                     BugReporterContext &BRC,
                                     PathSensitiveBugReport &BR) {
  using lifetime_modeling::isDeallocated;
  const ExplodedNode *Pred = N->getFirstPred();
  if (!Pred)
    return nullptr;

  if (!isDeallocated(N->getState(), SourceRegion) ||
      isDeallocated(Pred->getState(), SourceRegion))
    return nullptr;

  const Stmt *S = N->getStmtForDiagnostics();
  if (!S)
    return nullptr;

  PathDiagnosticLocation Pos = PathDiagnosticLocation::createEnd(
      S, BRC.getSourceManager(), N->getStackFrame());
  return std::make_shared<PathDiagnosticEventPiece>(
      Pos,
      (llvm::Twine("'") + SourceRegion->getString() + "' is destroyed here")
          .str(),
      true);
}

void ento::registerDanglingPtrDeref(CheckerManager &Mgr) {
  Mgr.registerChecker<DanglingPtrDeref>();
}

bool ento::shouldRegisterDanglingPtrDeref(const CheckerManager &Mgr) {
  return true;
}
