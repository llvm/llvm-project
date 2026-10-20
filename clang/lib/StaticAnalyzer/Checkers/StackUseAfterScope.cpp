#include "LifetimeModeling.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class StackUseAfterScope : public Checker<check::Location, check::PostCall> {
public:
  void checkLocation(SVal Loc, bool IsLoad, const Stmt *S,
                     CheckerContext &C) const;
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void reportUseAfterScope(const MemRegion *Region, const Stmt *S,
                           ExplodedNode *N, CheckerContext &C) const;
  const BugType BugMsg{this, "StackUseAfterScope", "LifetimeBound"};
};

class StackUseAfterScopeBRVisitor : public BugReporterVisitor {
  const MemRegion *SourceRegion;

public:
  explicit StackUseAfterScopeBRVisitor(const MemRegion *Source)
      : SourceRegion(Source) {}

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    ID.AddPointer(SourceRegion);
  }

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override;
};

} // namespace

void StackUseAfterScope::checkLocation(SVal Loc, bool IsLoad, const Stmt *S,
                                     CheckerContext &C) const {
  ProgramStateRef State = C.getState();

  if (const MemRegion *LocRegion = Loc.getAsRegion()) {
    if (lifetime_modeling::isDeallocated(State, LocRegion)) {
      if (ExplodedNode *N = C.generateNonFatalErrorNode(State))
        reportUseAfterScope(LocRegion, S, N, C);
    }
  }
}

void StackUseAfterScope::checkPostCall(const CallEvent &Call,
                                     CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  // Only check calls arguments if it is not inlined by the engine. In case a
  // function is inlined checkLocation handles any dereference in its body.
  if (C.wasInlined)
    return;

  for (unsigned Idx = 0; Idx < Call.getNumArgs(); Idx++) {
    if (const MemRegion *ArgRegion = Call.getArgSVal(Idx).getAsRegion())
      if (lifetime_modeling::isDeallocated(State, ArgRegion))
        if (ExplodedNode *N = C.generateNonFatalErrorNode())
          reportUseAfterScope(ArgRegion, Call.getArgExpr(Idx), N, C);
  }
}

void StackUseAfterScope::reportUseAfterScope(const MemRegion *Region,
                                           const Stmt *S, ExplodedNode *N,
                                           CheckerContext &C) const {
  ProgramStateRef ReportedState =
      lifetime_modeling::markAsReported(N->getState(), Region);
  if (!ReportedState)
    return;

  auto BR = std::make_unique<PathSensitiveBugReport>(
      BugMsg,
      (llvm::Twine("Use of ") + lifetime_modeling::getRegionName(Region) +
       " after its lifetime ended."),
      N);
  BR->addVisitor<StackUseAfterScopeBRVisitor>(Region);
  if (S) {
    if (const Expr *DerefExpr = bugreporter::getDerefExpr(S))
      bugreporter::trackExpressionValue(N, DerefExpr, *BR);
  }
  C.addTransition(ReportedState, N);
  C.emitReport(std::move(BR));
}

PathDiagnosticPieceRef
StackUseAfterScopeBRVisitor::VisitNode(const ExplodedNode *N,
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
      (lifetime_modeling::getRegionName(SourceRegion) +
       llvm::Twine(" is destroyed here"))
          .str(),
      true);
}

void ento::registerStackUseAfterScope(CheckerManager &Mgr) {
  Mgr.registerChecker<StackUseAfterScope>();
}

bool ento::shouldRegisterStackUseAfterScope(const CheckerManager &Mgr) {
  return true;
}
