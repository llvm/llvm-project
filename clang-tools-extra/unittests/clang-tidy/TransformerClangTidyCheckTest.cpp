//===---- TransformerClangTidyCheckTest.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../clang-tidy/utils/TransformerClangTidyCheck.h"
#include "ClangTidyTest.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Refactoring/RangeSelector.h"
#include "clang/Tooling/Refactoring/Stencil.h"
#include "clang/Tooling/Refactoring/Transformer.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace utils {
namespace {
using tooling::RewriteRule;

// Invert the code of an if-statement, while maintaining its semantics.
RewriteRule invertIf() {
  using namespace ::clang::ast_matchers;
  using tooling::change;
  using tooling::node;
  using tooling::statement;
  using tooling::text;
  using tooling::stencil::cat;

  StringRef C = "C", T = "T", E = "E";
  RewriteRule Rule = tooling::makeRule(
      ifStmt(hasCondition(expr().bind(C)), hasThen(stmt().bind(T)),
             hasElse(stmt().bind(E))),
      change(
          statement(RewriteRule::RootID),
          cat("if(!(", node(C), ")) ", statement(E), " else ", statement(T))),
      text("negate condition and reverse `then` and `else` branches"));
  return Rule;
}

class IfInverterCheck : public TransformerClangTidyCheck {
public:
  IfInverterCheck(StringRef Name, ClangTidyContext *Context)
      : TransformerClangTidyCheck(invertIf(), Name, Context) {}
};

// Basic test of using a rewrite rule as a ClangTidy.
TEST(TransformerClangTidyCheckTest, Basic) {
  const std::string Input = R"cc(
    void log(const char* msg);
    void foo() {
      if (10 > 1.0)
        log("oh no!");
      else
        log("ok");
    }
  )cc";
  const std::string Expected = R"(
    void log(const char* msg);
    void foo() {
      if(!(10 > 1.0)) log("ok"); else log("oh no!");
    }
  )";
  EXPECT_EQ(Expected, test::runCheckOnCode<IfInverterCheck>(Input));
}
} // namespace
} // namespace utils
} // namespace tidy
} // namespace clang
