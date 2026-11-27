#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"

// This barebones plugin is used by clang/test/Analysis/checker-plugins.c
// to test dependency handling among checkers loaded from plugins.

using namespace clang;
using namespace ento;

namespace {
struct Dependency : public Checker<check::BeginFunction> {
  void checkBeginFunction(CheckerContext &Ctx) const {}
};
struct DependendentChecker : public Checker<check::BeginFunction> {
  void checkBeginFunction(CheckerContext &Ctx) const {}
};
} // end anonymous namespace

// Register plugin!
extern "C" void clang_registerCheckers(CheckerRegistry &Registry) {
  Registry.addChecker<Dependency>("example.Dependency", "MockDescription");
  Registry.addChecker<DependendentChecker>("example.DependendentChecker",
                                           "MockDescription");

  Registry.addDependency("example.DependendentChecker", "example.Dependency");
}

extern "C" const char clang_analyzerAPIVersionString[] =
    CLANG_ANALYZER_API_VERSION_STRING;
