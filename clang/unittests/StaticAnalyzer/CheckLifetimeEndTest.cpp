//===- unittests/StaticAnalyzer/CheckLifetimeEndTest.cpp ------------------===//
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
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ento;

REGISTER_TRAIT_WITH_PROGRAMSTATE(TestLifetimeEndReportCountTrait, unsigned)

namespace {

class LifetimeEndReporter : public Checker<check::LifetimeEnd> {
  const BugType LifetimeEndNode{this, "LifetimeEndReporter"};

public:
  void checkLifetimeEnd(const VarDecl *D, CheckerContext &C) const {
    ProgramStateRef State = C.getState();
    // Intentionally add a unique number to each report to avoid deduplication.
    unsigned Count = State->get<TestLifetimeEndReportCountTrait>();
    State = State->set<TestLifetimeEndReportCountTrait>(Count + 1);
    auto Description = llvm::formatv("{0} LIFETIME END {1}",
                                     D->getDeclName().getAsString(), Count);

    ExplodedNode *Node = C.generateNonFatalErrorNode(State);
    EXPECT_TRUE(Node != nullptr);

    auto Report = std::make_unique<PathSensitiveBugReport>(
        LifetimeEndNode, Description.str(), Node);
    C.emitReport(std::move(Report));
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
  EXPECT_EQ(Diags, "test.LifetimeEndReporter: i LIFETIME END 0\n");
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

TEST(CheckLifetimeEnd, NonTrivialDtor) {
  constexpr auto Code = R"(
 struct A {
   ~A() {}
 };
 void foo() {
   A a;
 }
   )";
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addLifetimeEndReporter>(
      Code, EnableLifetimeArgs, Diags, /*OnlyEmitWarnings=*/true));
  EXPECT_EQ(Diags, "test.LifetimeEndReporter: a LIFETIME END 0\n");
}

TEST(CheckLifetimeEnd, MultipleVariablesAndNestedScopes) {
  constexpr auto Code = R"(
void foo() {
  int a = 0;
  int b = 0;
  {
    int c = 0, d = 0;
  }
}
  )";

  std::string Diags;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addLifetimeEndReporter>(
      Code, EnableLifetimeArgs, Diags, /*OnlyEmitWarnings=*/true));
  EXPECT_EQ(Diags, "test.LifetimeEndReporter: a LIFETIME END 3\n"
                   "test.LifetimeEndReporter: b LIFETIME END 2\n"
                   "test.LifetimeEndReporter: c LIFETIME END 1\n"
                   "test.LifetimeEndReporter: d LIFETIME END 0\n");
}

TEST(CheckLifetimeEnd, LocalStaticVariable) {
  constexpr auto Code = R"(
void foo() {
  static int i = 0;
  int j = 0;
}
  )";

  std::string Diags;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addLifetimeEndReporter>(
      Code, EnableLifetimeArgs, Diags, /*OnlyEmitWarnings=*/true));
  EXPECT_EQ(Diags, "test.LifetimeEndReporter: j LIFETIME END 0\n");
}

TEST(CheckLifetimeEnd, GlobalVariable) {
  constexpr auto Code = R"(
int g = 0;
void foo() {
  int i = 0;
}
  )";

  std::string Diags;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addLifetimeEndReporter>(
      Code, EnableLifetimeArgs, Diags, /*OnlyEmitWarnings=*/true));
  EXPECT_EQ(Diags, "test.LifetimeEndReporter: i LIFETIME END 0\n");
}

TEST(CheckLifetimeEnd, LoopBodyVariable) {
  constexpr auto Code = R"(
void foo() {
  while (true) {
    int i = 0;
    break;
  }
}
  )";

  std::string Diags;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addLifetimeEndReporter>(
      Code, EnableLifetimeArgs, Diags, /*OnlyEmitWarnings=*/true));
  EXPECT_EQ(Diags, "test.LifetimeEndReporter: i LIFETIME END 0\n");
}

TEST(CheckLifetimeEnd, ForLoopInductionVariable) {
  constexpr auto Code = R"(
void foo() {
  for (int i = 0; i < 2; i++) {
    int j = 0;
    {
       int nested = 0;
    }
    ++j;
  }
}
  )";

  std::string Diags;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addLifetimeEndReporter>(
      Code, EnableLifetimeArgs, Diags, /*OnlyEmitWarnings=*/true));
  EXPECT_EQ(Diags, "test.LifetimeEndReporter: i LIFETIME END 4\n"
                   "test.LifetimeEndReporter: j LIFETIME END 1\n"
                   "test.LifetimeEndReporter: j LIFETIME END 3\n"
                   "test.LifetimeEndReporter: nested LIFETIME END 0\n"
                   "test.LifetimeEndReporter: nested LIFETIME END 2\n");
}

TEST(CheckLifetimeEnd, LifetimeExtendedTemporary) {
  constexpr auto Code = R"(
void foo() {
  const int& r = 42;
}
  )";

  std::string Diags;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addLifetimeEndReporter>(
      Code, EnableLifetimeArgs, Diags, /*OnlyEmitWarnings=*/true));
  EXPECT_EQ(Diags, "test.LifetimeEndReporter: r LIFETIME END 0\n");
}

} // namespace
