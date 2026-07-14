#include "LifetimeModeling.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"

using namespace clang;
using namespace ento;

namespace {
class UseAfterLifetimeEnd : public Checker<check::EndFunction> {
public:
  void reportDanglingSource(const MemRegion *Source, SVal Val, ExplodedNode *N,
                            CheckerContext &C) const;
  void checkEndFunction(const ReturnStmt *RS, CheckerContext &C) const;
  const BugType BugMsg{this, "UseAfterLifetimeEnd", "LifetimeBound"};
};

class UseAfterLifetimeEndBRVisitor : public BugReporterVisitor {
  SVal BoundVal;
  const MemRegion *SourceRegion;

public:
  explicit UseAfterLifetimeEndBRVisitor(SVal Val, const MemRegion *Source)
      : BoundVal(Val), SourceRegion(Source) {}

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    static int X = 0;
    ID.AddPointer(&X);
    BoundVal.Profile(ID);
    SourceRegion->Profile(ID);
  }

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override;
  PathDiagnosticPieceRef getEndPath(BugReporterContext &BRC,
                                    const ExplodedNode *N,
                                    PathSensitiveBugReport &BR) override;
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
      reportDanglingSource(R, RetVal, N, C);
  }
}

static SourceRange getRegionDeclRange(const MemRegion *Source) {
  if (const auto *VR = dyn_cast<VarRegion>(Source))
    return VR->getDecl()->getSourceRange();
  return SourceRange();
}

void UseAfterLifetimeEnd::reportDanglingSource(const MemRegion *Source,
                                               SVal Val, ExplodedNode *N,
                                               CheckerContext &C) const {
  auto BR = std::make_unique<PathSensitiveBugReport>(
      BugMsg,
      (llvm::Twine("Returning value bound to '") + Source->getString() +
       "' that will go out of scope"),
      N);

  if (SourceRange Range = getRegionDeclRange(Source); Range.isValid())
    BR->addRange(Range);

  BR->addVisitor<UseAfterLifetimeEndBRVisitor>(Val, Source);
  C.emitReport(std::move(BR));
}

PathDiagnosticPieceRef
UseAfterLifetimeEndBRVisitor::VisitNode(const ExplodedNode *N,
                                        BugReporterContext &BRC,
                                        PathSensitiveBugReport &BR) {
  const ExplodedNode *Pred = N->getFirstPred();
  if (!Pred)
    return nullptr;

  if (!lifetime_modeling::isBoundToLifetimeSource(N->getState(), BoundVal) ||
      lifetime_modeling::isBoundToLifetimeSource(Pred->getState(), BoundVal))
    return nullptr;

  const Stmt *S = N->getStmtForDiagnostics();
  if (!S)
    return nullptr;

  PathDiagnosticLocation Pos(S, BRC.getSourceManager(), N->getStackFrame());
  auto Piece = std::make_shared<PathDiagnosticEventPiece>(
      Pos,
      (llvm::Twine("Value bound to '") + SourceRegion->getString() + "' here")
          .str(),
      true);

  if (SourceRange Range = getRegionDeclRange(SourceRegion); Range.isValid())
    Piece->addRange(Range);

  return Piece;
}

PathDiagnosticPieceRef
UseAfterLifetimeEndBRVisitor::getEndPath(BugReporterContext &BRC,
                                         const ExplodedNode *N,
                                         PathSensitiveBugReport &BR) {
  const Stmt *S = N->getStmtForDiagnostics();
  if (!S)
    return nullptr;

  PathDiagnosticLocation Pos(S, BRC.getSourceManager(), N->getStackFrame());
  auto Piece = std::make_shared<PathDiagnosticEventPiece>(
      Pos,
      llvm::Twine(("Lifetime of '") + SourceRegion->getString() +
                  "' ended here")
          .str(),
      true);

  if (SourceRange Range = getRegionDeclRange(SourceRegion); Range.isValid())
    Piece->addRange(Range);

  return Piece;
}

void ento::registerUseAfterLifetimeEnd(CheckerManager &Mgr) {
  Mgr.registerChecker<UseAfterLifetimeEnd>();
}

bool ento::shouldRegisterUseAfterLifetimeEnd(const CheckerManager &Mgr) {
  return true;
}
