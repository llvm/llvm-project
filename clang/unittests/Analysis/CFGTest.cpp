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

// Constructing a CFG for a range-based for over a dependent type fails (but
// should not crash).
TEST(CFG, RangeBasedForOverDependentType) {
  const char *Code = "class Foo;\n"
                     "template <typename T>\n"
                     "void f(const T &Range) {\n"
                     "  for (const Foo *TheFoo : Range) {\n"
                     "  }\n"
                     "}\n";

  class CFGCallback : public ast_matchers::MatchFinder::MatchCallback {
  public:
    bool SawFunctionBody = false;

    void run(const ast_matchers::MatchFinder::MatchResult &Result) override {
      const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
      Stmt *Body = Func->getBody();
      if (!Body)
        return;
      SawFunctionBody = true;
      std::unique_ptr<CFG> cfg =
          CFG::buildCFG(nullptr, Body, Result.Context, CFG::BuildOptions());
      EXPECT_EQ(nullptr, cfg);
    }
  } Callback;

  ast_matchers::MatchFinder Finder;
  Finder.addMatcher(ast_matchers::functionDecl().bind("func"), &Callback);
  std::unique_ptr<tooling::FrontendActionFactory> Factory(
      tooling::newFrontendActionFactory(&Finder));
  std::vector<std::string> Args = {"-std=c++11", "-fno-delayed-template-parsing"};
  ASSERT_TRUE(tooling::runToolOnCodeWithArgs(Factory->create(), Code, Args));
  EXPECT_TRUE(Callback.SawFunctionBody);
}

} // namespace
} // namespace analysis
} // namespace clang
