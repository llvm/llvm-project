#include "LifetimeModeling.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"
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
  PathDiagnosticPieceRef getEndPath(const ExplodedNode *N,
                                    BugReporterContext &BRC,
                                    PathSensitiveBugReport &BR) override;
  PathDiagnosticPieceRef createSourcePiece(const ExplodedNode *N,
                                           BugReporterContext &BRC,
                                           StringRef Message) const;
};

} // namespace

static const Expr *getLifetimeBoundArg(const Expr *RetExpr) {
  const CallExpr *Expr = dyn_cast_or_null<CallExpr>(RetExpr);
  if (!Expr)
    return nullptr;
  const FunctionDecl *FD = Expr->getDirectCallee();
  if (!FD)
    return nullptr;

  for (const ParmVarDecl *PVD : FD->parameters()) {
    if (PVD->hasAttr<LifetimeBoundAttr>()) {
      unsigned Idx = PVD->getFunctionScopeIndex();
      if (Idx < Expr->getNumArgs())
        return Expr->getArg(Idx);
    }
  }
  return nullptr;
}

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
  if (const auto *VR = dyn_cast_or_null<VarRegion>(Source))
    return VR->getDecl()->getSourceRange();
  return SourceRange();
}

void UseAfterLifetimeEnd::reportDanglingSource(const MemRegion *Source,
                                               SVal RetVal, ExplodedNode *N,
                                               CheckerContext &C) const {
  auto BR = std::make_unique<PathSensitiveBugReport>(
      BugMsg,
      (llvm::Twine("Returning value bound to ") +
       lifetime_modeling::getRegionName(Source) + " that will go out of scope"),
      N);

  if (SourceRange Range = getRegionDeclRange(Source); Range.isValid())
    BR->addRange(Range);

  BR->addVisitor<UseAfterLifetimeEndBRVisitor>(RetVal, Source);
  bugreporter::trackStoredValue(RetVal, Source, *BR);
  C.emitReport(std::move(BR));
}

PathDiagnosticPieceRef UseAfterLifetimeEndBRVisitor::createSourcePiece(
    const ExplodedNode *N, BugReporterContext &BRC, StringRef Message) const {
  const Stmt *S = N->getStmtForDiagnostics();
  if (!S)
    return nullptr;

  const Expr *RetExpr = dyn_cast_or_null<Expr>(S);
  const Expr *Arg = getLifetimeBoundArg(RetExpr);

  PathDiagnosticLocation Pos;

  if (Arg != nullptr)
    Pos =
        PathDiagnosticLocation(Arg, BRC.getSourceManager(), N->getStackFrame());
  else
    Pos = PathDiagnosticLocation(S, BRC.getSourceManager(), N->getStackFrame());

  auto Note = std::make_shared<PathDiagnosticEventPiece>(Pos, Message, true);
  if (SourceRange Range = getRegionDeclRange(SourceRegion); Range.isValid())
    Note->addRange(Range);

  return Note;
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

  auto Piece = createSourcePiece(
      N, BRC,
      (llvm::Twine("Value's lifetime bound to the lifetime of ") +
       lifetime_modeling::getRegionName(SourceRegion) + " here")
          .str());
  return Piece;
}

PathDiagnosticPieceRef
UseAfterLifetimeEndBRVisitor::getEndPath(const ExplodedNode *N,
                                         BugReporterContext &BRC,
                                         PathSensitiveBugReport &BR) {
  auto Piece = createSourcePiece(
      N, BRC,
      (llvm::Twine("Lifetime of ") +
       lifetime_modeling::getRegionName(SourceRegion) + " ended here")
          .str());
  return Piece;
}

void ento::registerUseAfterLifetimeEnd(CheckerManager &Mgr) {
  Mgr.registerChecker<UseAfterLifetimeEnd>();
}

bool ento::shouldRegisterUseAfterLifetimeEnd(const CheckerManager &Mgr) {
  return true;
}
