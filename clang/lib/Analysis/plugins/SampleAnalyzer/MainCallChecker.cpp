#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"

// This simple plugin is used by clang/test/Analysis/checker-plugins.c
// to test the use of a checker that is defined in a plugin.

using namespace clang;
using namespace ento;

namespace {
class MainCallChecker : public Checker<check::PreStmt<CallExpr>> {

  const BugType BT{this, "call to main", "example analyzer plugin"};

public:
  void checkPreStmt(const CallExpr *CE, CheckerContext &C) const;
};
} // end anonymous namespace

void MainCallChecker::checkPreStmt(const CallExpr *CE,
                                   CheckerContext &C) const {
  const Expr *Callee = CE->getCallee();
  const FunctionDecl *FD = C.getSVal(Callee).getAsFunctionDecl();

  if (!FD)
    return;

  // Get the name of the callee.
  IdentifierInfo *II = FD->getIdentifier();
  if (!II) // if no identifier, not a simple C function
    return;

  if (II->isStr("main")) {
    ExplodedNode *N = C.generateErrorNode();
    if (!N)
      return;

    auto report =
        std::make_unique<PathSensitiveBugReport>(BT, BT.getDescription(), N);
    report->addRange(Callee->getSourceRange());
    C.emitReport(std::move(report));
  }
}

// Register plugin!
extern "C" void clang_registerCheckers(CheckerRegistry &Registry) {
  Registry.addChecker<MainCallChecker>("example.MainCallChecker",
                                       "Example Description");
}

extern "C" const char clang_analyzerAPIVersionString[] =
    CLANG_ANALYZER_API_VERSION_STRING;
