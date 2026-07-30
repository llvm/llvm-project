#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"

using namespace clang;
using namespace ento;

// This barebones plugin is used by clang/test/Analysis/checker-plugins.c
// to test option handling on checkers loaded from plugins.

namespace {
struct MyChecker : public Checker<check::BeginFunction> {
  void checkBeginFunction(CheckerContext &Ctx) const {}
};

void registerMyChecker(CheckerManager &Mgr) {
  MyChecker *Checker = Mgr.registerChecker<MyChecker>();
  llvm::outs() << "Example option is set to "
               << (Mgr.getAnalyzerOptions().getCheckerBooleanOption(
                       Checker, "ExampleOption")
                       ? "true"
                       : "false")
               << '\n';
}

bool shouldRegisterMyChecker(const CheckerManager &mgr) { return true; }

} // end anonymous namespace

// Register plugin!
extern "C" void clang_registerCheckers(CheckerRegistry &Registry) {
  Registry.addChecker(registerMyChecker, shouldRegisterMyChecker,
                      "example.MyChecker", "Example Description");

  Registry.addCheckerOption(/*OptionType*/ "bool",
                            /*CheckerFullName*/ "example.MyChecker",
                            /*OptionName*/ "ExampleOption",
                            /*DefaultValStr*/ "false",
                            /*Description*/ "This is an example checker opt.",
                            /*DevelopmentStage*/ "released");
}

extern "C" const char clang_analyzerAPIVersionString[] =
    CLANG_ANALYZER_API_VERSION_STRING;
