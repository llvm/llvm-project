#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Checkers/LifetimeModeling.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"
#include "clang/StaticAnalyzer/Core/Checker.h"

using namespace clang;
using namespace ento;

namespace {
class UseAfterLifetimeEnd : public Checker<check::EndFunction> {
public:
  class UseAfterLifetimeEndBRVisitor : public BugReporterVisitor {
    SVal BoundRegion;
    const MemRegion *SourceRegion;

  public:
    UseAfterLifetimeEndBRVisitor(SVal Region, const MemRegion *Source)
        : BoundRegion(Region), SourceRegion(Source) {}

    void Profile(llvm::FoldingSetNodeID &ID) const override {
      static int X = 0;
      ID.AddPointer(&X);
      BoundRegion.Profile(ID);
      SourceRegion->Profile(ID);
    }

    PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                     BugReporterContext &BRC,
                                     PathSensitiveBugReport &BR) override;
    PathDiagnosticPieceRef getEndPath(BugReporterContext &BRC,
                                      const ExplodedNode *N,
                                      PathSensitiveBugReport &BR) override;
  };

  void reportDanglingSource(const MemRegion *Source, SVal Val, ExplodedNode *N,
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

  std::vector<const MemRegion *> RetValRegion =
      lifetime_modeling::checkReturnedBorrower(RetVal, State, C);
  for (const auto *Region : RetValRegion) {
    if (!N)
      N = C.generateNonFatalErrorNode();
    if (!N)
      return;

    reportDanglingSource(Region, RetVal, N, C);
  }
}

void UseAfterLifetimeEnd::reportDanglingSource(const MemRegion *Source,
                                               SVal Val, ExplodedNode *N,
                                               CheckerContext &C) const {
  auto BR = std::make_unique<PathSensitiveBugReport>(
      BugMsg,
      (llvm::Twine("Returning value bound to '") + Source->getString() +
       "' that will go out of scope"),
      N);
  BR->addVisitor(std::make_unique<UseAfterLifetimeEndBRVisitor>(Val, Source));
  C.emitReport(std::move(BR));
}

PathDiagnosticPieceRef
UseAfterLifetimeEnd::UseAfterLifetimeEndBRVisitor::VisitNode(
    const ExplodedNode *N, BugReporterContext &BRC,
    PathSensitiveBugReport &BR) {
  if (!lifetime_modeling::isBoundToLifetimeSource(BoundRegion, N->getState()) ||
      lifetime_modeling::isBoundToLifetimeSource(BoundRegion,
                                                 N->getFirstPred()->getState()))
    return nullptr;

  const Stmt *S = N->getStmtForDiagnostics();
  if (!S)
    return nullptr;

  SmallString<256> Str;
  llvm::raw_svector_ostream OS(Str);
  OS << "Value bound to '" << SourceRegion->getString() << "' here";
  PathDiagnosticLocation Pos(S, BRC.getSourceManager(), N->getStackFrame());
  return std::make_shared<PathDiagnosticEventPiece>(Pos, OS.str(), true);
}

PathDiagnosticPieceRef
UseAfterLifetimeEnd::UseAfterLifetimeEndBRVisitor::getEndPath(
    BugReporterContext &BRC, const ExplodedNode *N,
    PathSensitiveBugReport &BR) {
  const Stmt *S = N->getStmtForDiagnostics();
  if (!S)
    return nullptr;

  SmallString<256> Str;
  llvm::raw_svector_ostream OS(Str);
  OS << "Lifetime of '" << SourceRegion->getString() << "' ended here";
  PathDiagnosticLocation Pos(S, BRC.getSourceManager(), N->getStackFrame());
  return std::make_shared<PathDiagnosticEventPiece>(Pos, OS.str(), true);
}

void ento::registerUseAfterLifetimeEnd(CheckerManager &Mgr) {
  Mgr.registerChecker<UseAfterLifetimeEnd>();
}

bool ento::shouldRegisterUseAfterLifetimeEnd(const CheckerManager &Mgr) {
  return true;
}
