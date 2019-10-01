//===- unittests/Analysis/CFGTest.cpp - CFG tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/CFG.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

namespace clang {
namespace analysis {
namespace {

class BuildResult {
public:
  enum Status {
    ToolFailed,
    ToolRan,
    SawFunctionBody,
    BuiltCFG,
  };

  BuildResult(Status S, std::unique_ptr<CFG> Cfg = nullptr)
      : S(S), Cfg(std::move(Cfg)) {}

  Status getStatus() const { return S; }
  CFG *getCFG() const { return Cfg.get(); }

private:
  Status S;
  std::unique_ptr<CFG> Cfg;
};

class CFGCallback : public ast_matchers::MatchFinder::MatchCallback {
public:
  BuildResult TheBuildResult = BuildResult::ToolRan;

  void run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
    Stmt *Body = Func->getBody();
    if (!Body)
      return;
    TheBuildResult = BuildResult::SawFunctionBody;
    CFG::BuildOptions Options;
    Options.AddImplicitDtors = true;
    if (std::unique_ptr<CFG> Cfg =
            CFG::buildCFG(nullptr, Body, Result.Context, Options))
      TheBuildResult = {BuildResult::BuiltCFG, std::move(Cfg)};
  }
};

BuildResult BuildCFG(const char *Code) {
  CFGCallback Callback;

  ast_matchers::MatchFinder Finder;
  Finder.addMatcher(ast_matchers::functionDecl().bind("func"), &Callback);
  std::unique_ptr<tooling::FrontendActionFactory> Factory(
      tooling::newFrontendActionFactory(&Finder));
  std::vector<std::string> Args = {"-std=c++11", "-fno-delayed-template-parsing"};
  if (!tooling::runToolOnCodeWithArgs(Factory->create(), Code, Args))
    return BuildResult::ToolFailed;
  return std::move(Callback.TheBuildResult);
}

// Constructing a CFG for a range-based for over a dependent type fails (but
// should not crash).
TEST(CFG, RangeBasedForOverDependentType) {
  const char *Code = "class Foo;\n"
                     "template <typename T>\n"
                     "void f(const T &Range) {\n"
                     "  for (const Foo *TheFoo : Range) {\n"
                     "  }\n"
                     "}\n";
  EXPECT_EQ(BuildResult::SawFunctionBody, BuildCFG(Code).getStatus());
}

// Constructing a CFG containing a delete expression on a dependent type should
// not crash.
TEST(CFG, DeleteExpressionOnDependentType) {
  const char *Code = "template<class T>\n"
                     "void f(T t) {\n"
                     "  delete t;\n"
                     "}\n";
  EXPECT_EQ(BuildResult::BuiltCFG, BuildCFG(Code).getStatus());
}

// Constructing a CFG on a function template with a variable of incomplete type
// should not crash.
TEST(CFG, VariableOfIncompleteType) {
  const char *Code = "template<class T> void f() {\n"
                     "  class Undefined;\n"
                     "  Undefined u;\n"
                     "}\n";
  EXPECT_EQ(BuildResult::BuiltCFG, BuildCFG(Code).getStatus());
}

TEST(CFG, IsLinear) {
  auto expectLinear = [](bool IsLinear, const char *Code) {
    BuildResult B = BuildCFG(Code);
    EXPECT_EQ(BuildResult::BuiltCFG, B.getStatus());
    EXPECT_EQ(IsLinear, B.getCFG()->isLinear());
  };

  expectLinear(true,  "void foo() {}");
  expectLinear(true,  "void foo() { if (true) return; }");
  expectLinear(true,  "void foo() { if constexpr (false); }");
  expectLinear(false, "void foo(bool coin) { if (coin) return; }");
  expectLinear(false, "void foo() { for(;;); }");
  expectLinear(false, "void foo() { do {} while (true); }");
  expectLinear(true,  "void foo() { do {} while (false); }");
  expectLinear(true,  "void foo() { foo(); }"); // Recursion is not our problem.
}

} // namespace
} // namespace analysis
} // namespace clang
