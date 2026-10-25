#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"

#include <iostream>

using namespace clang;
using namespace ento;

namespace {
class StatsChecker
    : public Checker<check::EndAnalysis> {

public:
  void checkEndAnalysis(ExplodedGraph &G, BugReporter &B,
                        ExprEngine &Engine) const;

};
} // namespace

void StatsChecker::checkEndAnalysis(ExplodedGraph &G, BugReporter &B,
                                    ExprEngine &Engine) const
{
    std::cout << "END OF ANALYSIS" << std::endl;
    std::cout << "Number of exploded nodes created: "
              << G.size() + G.reclaimedNodes() << std::endl;
    std::cout << "Number of exploded nodes at the end: "
              << G.size() << std::endl;
}


void ento::registerStatsChecker(CheckerManager &mgr) {
  auto* Checker = mgr.registerChecker<StatsChecker>();
}

bool ento::shouldRegisterStatsChecker(const CheckerManager &mgr) {
  return true;
}
