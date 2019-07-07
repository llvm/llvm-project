//===- unittests/Analysis/CFGTest.cpp - CFG tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CFGBuildResult.h"
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

TEST(CFG, ConditionExpr) {
  const char *Code = R"(void f(bool A, bool B, bool C) {
                          if (A && B && C)
                            int x;
                        })";
  BuildResult Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());

  // [B5 (ENTRY)] -> [B4] -> [B3] -> [B2] -> [B1] -> [B0 (EXIT)]
  //                   \      \       \                 /
  //                    ------------------------------->

  CFG *cfg = Result.getCFG();

  auto GetBlock = [cfg] (unsigned Index) -> CFGBlock * {
    assert(Index < cfg->size());
    return *(cfg->begin() + Index);
  };

  EXPECT_EQ(GetBlock(1)->getLastCondition(), nullptr);
  // Unfortunately, we can't check whether the correct Expr was returned by
  // getLastCondition, because the lifetime of the AST ends by the time we
  // retrieve the CFG.

  //===--------------------------------------------------------------------===//

  Code = R"(void foo(int x, int y) {
              (void)(x + y);
            })";
  Result = BuildCFG(Code);
  EXPECT_EQ(BuildResult::BuiltCFG, Result.getStatus());

  // [B2 (ENTRY)] -> [B1] -> [B0 (EXIT)]

  cfg = Result.getCFG();
  EXPECT_EQ(GetBlock(1)->getLastCondition(), nullptr);
}

} // namespace
} // namespace analysis
} // namespace clang
