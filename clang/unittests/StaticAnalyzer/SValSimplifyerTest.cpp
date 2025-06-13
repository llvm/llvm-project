//===- unittests/StaticAnalyzer/SValSimplifyerTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ento;

static std::string toString(SVal V) {
  std::string Result;
  llvm::raw_string_ostream Stream(Result);
  V.dumpToStream(Stream);
  return Result;
}

static void replace(std::string &Content, StringRef Substr,
                    StringRef Replacement) {
  std::size_t Pos = 0;
  while ((Pos = Content.find(Substr, Pos)) != std::string::npos) {
    Content.replace(Pos, Substr.size(), Replacement);
    Pos += Replacement.size();
  }
}

namespace {

class SimplifyChecker : public Checker<check::PreCall> {
  const BugType Bug{this, "SimplifyChecker"};
  const CallDescription SimplifyCall{CDM::SimpleFunc, {"simplify"}, 1};

  void report(CheckerContext &C, const Expr *E, StringRef Description) const {
    PathDiagnosticLocation Loc(E->getExprLoc(), C.getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Description, Loc);
    C.emitReport(std::move(Report));
  }

public:
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const {
    if (!SimplifyCall.matches(Call))
      return;
    const Expr *Arg = Call.getArgExpr(0);
    SVal Val = C.getSVal(Arg);
    SVal SimplifiedVal = C.getSValBuilder().simplifySVal(C.getState(), Val);
    std::string Subject = toString(Val);
    std::string Simplified = toString(SimplifiedVal);
    std::string Message = (llvm::Twine{Subject} + " -> " + Simplified).str();
    report(C, Arg, Message);
  }
};
} // namespace

static void addSimplifyChecker(AnalysisASTConsumer &AnalysisConsumer,
                               AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"SimplifyChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<SimplifyChecker>("SimplifyChecker", "EmptyDescription",
                                         "EmptyDocsUri");
  });
}

static void runThisCheckerOnCode(const std::string &Code, std::string &Diags) {
  ASSERT_TRUE(runCheckerOnCode<addSimplifyChecker>(Code, Diags,
                                                   /*OnlyEmitWarnings=*/true));
  ASSERT_FALSE(Diags.empty());
  ASSERT_EQ(Diags.back(), '\n');
  Diags.pop_back();
}

namespace {

TEST(SValSimplifyerTest, LHSConstrainedNullPtrDiff) {
  constexpr auto Code = R"cpp(
template <class T> void simplify(T);
void LHSConstrainedNullPtrDiff(char *p, char *q) {
  int diff = p - q;
  if (!p)
    simplify(diff);
})cpp";

  std::string Diags;
  runThisCheckerOnCode(Code, Diags);
  replace(Diags, "(reg_$0<char * p>)", "reg_p");
  replace(Diags, "(reg_$1<char * q>)", "reg_q");
  // This should not be simplified to "Unknown".
  EXPECT_EQ(Diags, "SimplifyChecker: reg_p - reg_q -> 0U - reg_q");
}

} // namespace
