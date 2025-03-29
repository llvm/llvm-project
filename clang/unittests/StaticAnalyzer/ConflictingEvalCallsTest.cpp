//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ento;

namespace {
class EvalCallBase : public Checker<eval::Call> {
  const CallDescription Foo = {CDM::SimpleFunc, {"foo"}, 0};

public:
  bool evalCall(const CallEvent &Call, CheckerContext &C) const {
    return Foo.matches(Call);
  }
};

class EvalCallFoo1 : public EvalCallBase {};
class EvalCallFoo2 : public EvalCallBase {};
void addEvalFooCheckers(AnalysisASTConsumer &AnalysisConsumer,
                        AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.EvalFoo1", true},
                                {"test.EvalFoo2", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<EvalCallFoo1>("test.EvalFoo1", "EmptyDescription",
                                      "EmptyDocsUri");
    Registry.addChecker<EvalCallFoo2>("test.EvalFoo2", "EmptyDescription",
                                      "EmptyDocsUri");
  });
}
} // namespace

TEST(EvalCall, DetectConflictingEvalCalls) {
#ifdef NDEBUG
  GTEST_SKIP() << "This test is only available for debug builds.";
#endif
  const std::string Code = R"code(
    void foo();
    void top() {
      foo(); // crash
    }
  )code";
  constexpr auto Regex =
      "The 'foo\\(\\)' call has been already evaluated by the test\\.EvalFoo1 "
      "checker, while the test\\.EvalFoo2 checker also tried to evaluate the "
      "same call\\. At most one checker supposed to evaluate a call\\.";
  ASSERT_DEATH(runCheckerOnCode<addEvalFooCheckers>(Code), Regex);
}
