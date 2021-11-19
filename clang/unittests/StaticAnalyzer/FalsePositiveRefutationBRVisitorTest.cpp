//===- unittests/StaticAnalyzer/FalsePositiveRefutationBRVisitorTest.cpp --===//
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

class FalsePositiveGenerator : public Checker<eval::Call> {
  using Self = FalsePositiveGenerator;
  const BuiltinBug FalsePositiveGeneratorBug{this, "FalsePositiveGenerator"};
  using HandlerFn = bool (Self::*)(const CallEvent &Call,
                                   CheckerContext &) const;
  CallDescriptionMap<HandlerFn> Callbacks = {
      {{"reachedWithContradiction", 0}, &Self::reachedWithContradiction},
      {{"reachedWithNoContradiction", 0}, &Self::reachedWithNoContradiction},
      {{"reportIfCanBeTrue", 1}, &Self::reportIfCanBeTrue},
  };

  bool report(CheckerContext &C, ProgramStateRef State,
              StringRef Description) const {
    ExplodedNode *Node = C.generateNonFatalErrorNode(State);
    if (!Node)
      return false;

    auto Report = std::make_unique<PathSensitiveBugReport>(
        FalsePositiveGeneratorBug, Description, Node);
    C.emitReport(std::move(Report));
    return true;
  }

  bool reachedWithNoContradiction(const CallEvent &, CheckerContext &C) const {
    return report(C, C.getState(), "REACHED_WITH_NO_CONTRADICTION");
  }

  bool reachedWithContradiction(const CallEvent &, CheckerContext &C) const {
    return report(C, C.getState(), "REACHED_WITH_CONTRADICTION");
  }

  // Similar to ExprInspectionChecker::analyzerEval except it emits warning only
  // if the argument can be true. The report emits the report in the state where
  // the assertion true.
  bool reportIfCanBeTrue(const CallEvent &Call, CheckerContext &C) const {
    // A specific instantiation of an inlined function may have more constrained
    // values than can generally be assumed. Skip the check.
    if (C.getPredecessor()->getLocationContext()->getStackFrame()->getParent())
      return false;

    SVal AssertionVal = Call.getArgSVal(0);
    if (AssertionVal.isUndef())
      return false;

    ProgramStateRef State = C.getPredecessor()->getState();
    ProgramStateRef StTrue;
    std::tie(StTrue, std::ignore) =
        State->assume(AssertionVal.castAs<DefinedOrUnknownSVal>());
    if (StTrue)
      return report(C, StTrue, "CAN_BE_TRUE");
    return false;
  }

public:
  bool evalCall(const CallEvent &Call, CheckerContext &C) const {
    if (const HandlerFn *Callback = Callbacks.lookup(Call))
      return (this->*(*Callback))(Call, C);
    return false;
  }
};

void addFalsePositiveGenerator(AnalysisASTConsumer &AnalysisConsumer,
                               AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.FalsePositiveGenerator", true},
                                {"debug.ViewExplodedGraph", false}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<FalsePositiveGenerator>(
        "test.FalsePositiveGenerator", "EmptyDescription", "EmptyDocsUri");
  });
}

class FalsePositiveRefutationBRVisitorTestBase : public testing::Test {
public:
  void SetUp() override {
#ifndef LLVM_WITH_Z3
    GTEST_SKIP() << "Requires the LLVM_ENABLE_Z3_SOLVER cmake option.";
#endif
  }
};

// C++20 use constexpr below.
const std::vector<std::string> LazyAssumeArgs{
    "-Xclang", "-analyzer-config", "-Xclang", "eagerly-assume=false"};
const std::vector<std::string> LazyAssumeAndCrossCheckArgs{
    "-Xclang", "-analyzer-config", "-Xclang", "eagerly-assume=false",
    "-Xclang", "-analyzer-config", "-Xclang", "crosscheck-with-z3=true"};

TEST_F(FalsePositiveRefutationBRVisitorTestBase, UnSatInTheMiddleNoReport) {
  constexpr auto Code = R"(
     void reachedWithContradiction();
     void reachedWithNoContradiction();
     void test(int x, int y) {
       if (x * y == 0)
         return;
       reachedWithNoContradiction();
       if (x == 0) {
         reachedWithContradiction();
         // x * y != 0  =>  x != 0 && y != 0  => contradict with x == 0
       }
     })";

  std::string Diags;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addFalsePositiveGenerator>(
      Code, LazyAssumeAndCrossCheckArgs, Diags, /*OnlyEmitWarnings=*/ true));
  EXPECT_EQ(Diags,
            "test.FalsePositiveGenerator: REACHED_WITH_NO_CONTRADICTION\n");
  // Single warning. The second report was invalidated by the visitor.

  // Without enabling the crosscheck-with-z3 both reports are displayed.
  std::string Diags2;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addFalsePositiveGenerator>(
      Code, LazyAssumeArgs, Diags2, /*OnlyEmitWarnings=*/ true));
  EXPECT_EQ(Diags2,
            "test.FalsePositiveGenerator: REACHED_WITH_NO_CONTRADICTION\n"
            "test.FalsePositiveGenerator: REACHED_WITH_CONTRADICTION\n");
}

TEST_F(FalsePositiveRefutationBRVisitorTestBase,
       UnSatAtErrorNodeWithNewSymbolNoReport) {
  constexpr auto Code = R"(
    void reportIfCanBeTrue(bool);
    void reachedWithNoContradiction();
    void test(int x, int y) {
      if (x * y == 0)
       return;
      // We know that 'x * y': {[MIN,-1], [1,MAX]}
      reachedWithNoContradiction();
      reportIfCanBeTrue(x == 0); // contradiction
      // The function introduces the 'x == 0' constraint in the ErrorNode which
      // leads to contradiction with the constraint of 'x * y'.
      // Note that the new constraint was bound to a new symbol 'x'.
    })";
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addFalsePositiveGenerator>(
      Code, LazyAssumeAndCrossCheckArgs, Diags, /*OnlyEmitWarnings=*/ true));
  EXPECT_EQ(Diags,
            "test.FalsePositiveGenerator: REACHED_WITH_NO_CONTRADICTION\n");
  // Single warning. The second report was invalidated by the visitor.

  // Without enabling the crosscheck-with-z3 both reports are displayed.
  std::string Diags2;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addFalsePositiveGenerator>(
      Code, LazyAssumeArgs, Diags2, /*OnlyEmitWarnings=*/ true));
  EXPECT_EQ(Diags2,
            "test.FalsePositiveGenerator: REACHED_WITH_NO_CONTRADICTION\n"
            "test.FalsePositiveGenerator: CAN_BE_TRUE\n");
}

TEST_F(FalsePositiveRefutationBRVisitorTestBase,
       UnSatAtErrorNodeDueToRefinedConstraintNoReport) {
  constexpr auto Code = R"(
    void reportIfCanBeTrue(bool);
    void reachedWithNoContradiction();
    void test(unsigned x, unsigned n) {
      if (n >= 1 && n <= 2) {
        if (x >= 3)
          return;
        // x: [0,2] and n: [1,2]
        int y = x + n; // y: '(x+n)' Which is in approximately between 1 and 4.

        // Registers the symbol 'y' with the constraint [1, MAX] in the true
        // branch.
        if (y > 0) {
          // Since the x: [0,2] and n: [1,2], the 'y' is indeed greater than
          // zero. If we emit a warning here, the constraints on the BugPath is
          // SAT. Therefore that report is NOT invalidated.
          reachedWithNoContradiction(); // 'y' can be greater than zero. OK

          // If we ask the analyzer whether the 'y' can be 5. It won't know,
          // therefore, the state will be created where the 'y' expression is 5.
          // Although, this assumption is false!
          // 'y' can not be 5 if the maximal value of both x and n is 2.
          // The BugPath which become UnSAT in the ErrorNode with a refined
          // constraint, should be invalidated.
          reportIfCanBeTrue(y == 5);
        }
      }
    })";

  std::string Diags;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addFalsePositiveGenerator>(
      Code, LazyAssumeAndCrossCheckArgs, Diags, /*OnlyEmitWarnings=*/ true));
  EXPECT_EQ(Diags,
            "test.FalsePositiveGenerator: REACHED_WITH_NO_CONTRADICTION\n");
  // Single warning. The second report was invalidated by the visitor.

  // Without enabling the crosscheck-with-z3 both reports are displayed.
  std::string Diags2;
  EXPECT_TRUE(runCheckerOnCodeWithArgs<addFalsePositiveGenerator>(
      Code, LazyAssumeArgs, Diags2, /*OnlyEmitWarnings=*/ true));
  EXPECT_EQ(Diags2,
            "test.FalsePositiveGenerator: REACHED_WITH_NO_CONTRADICTION\n"
            "test.FalsePositiveGenerator: CAN_BE_TRUE\n");
}

} // namespace
} // namespace ento
} // namespace clang
