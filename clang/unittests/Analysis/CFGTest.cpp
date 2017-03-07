//===- unittests/Analysis/CFGTest.cpp - CFG tests -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

enum BuildResult {
  ToolFailed,
  ToolRan,
  SawFunctionBody,
  BuiltCFG,
};

class CFGCallback : public ast_matchers::MatchFinder::MatchCallback {
public:
  BuildResult TheBuildResult = ToolRan;

  void run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
    Stmt *Body = Func->getBody();
    if (!Body)
      return;
    TheBuildResult = SawFunctionBody;
    CFG::BuildOptions Options;
    Options.AddImplicitDtors = true;
    if (CFG::buildCFG(nullptr, Body, Result.Context, Options))
        TheBuildResult = BuiltCFG;
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
    return ToolFailed;
  return Callback.TheBuildResult;
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
  EXPECT_EQ(SawFunctionBody, BuildCFG(Code));
}

// Constructing a CFG containing a delete expression on a dependent type should
// not crash.
TEST(CFG, DeleteExpressionOnDependentType) {
  const char *Code = "template<class T>\n"
                     "void f(T t) {\n"
                     "  delete t;\n"
                     "}\n";
  EXPECT_EQ(BuiltCFG, BuildCFG(Code));
}

// Constructing a CFG on a function template with a variable of incomplete type
// should not crash.
TEST(CFG, VariableOfIncompleteType) {
  const char *Code = "template<class T> void f() {\n"
                     "  class Undefined;\n"
                     "  Undefined u;\n"
                     "}\n";
  EXPECT_EQ(BuiltCFG, BuildCFG(Code));
}

} // namespace
} // namespace analysis
} // namespace clang
