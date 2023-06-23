//===- unittests/StaticAnalyzer/CheckLifetimeEndTest.cpp --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "Reusables.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "llvm/Config/llvm-config.h"
#include "gtest/gtest.h"

namespace clang {
namespace ento {
namespace {

class LifetimeEndReporter : public Checker<check::LifetimeEnd> {
  using Self = LifetimeEndReporter;
  const BugType LifetimeEndNode{this, "LifetimeEndReporter"};

  bool report(CheckerContext &C, StringRef Description) const {
    ExplodedNode *Node = C.generateNonFatalErrorNode(C.getState());
    if (!Node)
      return false;

    auto Report = std::make_unique<PathSensitiveBugReport>(LifetimeEndNode,
                                                           Description, Node);
    C.emitReport(std::move(Report));
    return true;
  }

public:
  void checkLifetimeEnd(const VarDecl *D, CheckerContext &C) const {
    if (auto II = D->getIdentifier())
      report(C, (II->getName() + " LIFETIME END").str());
  }
};

void addLifetimeEndReporter(AnalysisASTConsumer &AnalysisConsumer,
                            AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {
      {"test.LifetimeEndReporter", true},
  };
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<LifetimeEndReporter>(
        "test.LifetimeEndReporter", "EmptyDescription", "EmptyDocsUri");
  });
}

const std::vector<std::string> DisableLifetimeArgs{
    "-Xclang", "-analyzer-config", "-Xclang", "cfg-lifetime=false"};
const std::vector<std::string> EnableLifetimeArgs{
    "-Xclang", "-analyzer-config", "-Xclang", "cfg-lifetime=true"};

TEST(CheckLifetimeEnd, CFGLifetimeEnabled) {
  constexpr auto Code = R"(
void foo() {
  int i = 0;
}
  )";

  std::string Diags;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addLifetimeEndReporter>(
      Code, EnableLifetimeArgs, Diags, /*OnlyEmitWarnings=*/true));
  EXPECT_EQ(Diags, "test.LifetimeEndReporter: i LIFETIME END\n");
}

TEST(CheckLifetimeEnd, CFGLifetimeDisabled) {
  constexpr auto Code = R"(
void foo() {
  int i = 0;
}
  )";

  std::string Diags;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addLifetimeEndReporter>(
      Code, DisableLifetimeArgs, Diags, /*OnlyEmitWarnings=*/true));
  EXPECT_TRUE(Diags.empty());
}

} // namespace
} // namespace ento
} // namespace clang
