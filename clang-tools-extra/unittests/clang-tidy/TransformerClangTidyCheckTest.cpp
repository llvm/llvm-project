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
using tooling::change;
using tooling::RewriteRule;
using tooling::text;
using tooling::stencil::cat;

// Invert the code of an if-statement, while maintaining its semantics.
RewriteRule invertIf() {
  using namespace ::clang::ast_matchers;
  using tooling::node;
  using tooling::statement;

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

// A trivial rewrite-rule generator that requires Objective-C code.
Optional<RewriteRule> needsObjC(const LangOptions &LangOpts,
                                const ClangTidyCheck::OptionsView &Options) {
  if (!LangOpts.ObjC)
    return None;
  return tooling::makeRule(clang::ast_matchers::functionDecl(),
                           change(cat("void changed() {}")), text("no message"));
}

class NeedsObjCCheck : public TransformerClangTidyCheck {
public:
  NeedsObjCCheck(StringRef Name, ClangTidyContext *Context)
      : TransformerClangTidyCheck(needsObjC, Name, Context) {}
};

// Verify that the check only rewrites the code when the input is Objective-C.
TEST(TransformerClangTidyCheckTest, DisableByLang) {
  const std::string Input = "void log() {}";
  EXPECT_EQ(Input,
            test::runCheckOnCode<NeedsObjCCheck>(Input, nullptr, "input.cc"));

  EXPECT_EQ("void changed() {}",
            test::runCheckOnCode<NeedsObjCCheck>(Input, nullptr, "input.mm"));
}

// A trivial rewrite rule generator that checks config options.
Optional<RewriteRule> noSkip(const LangOptions &LangOpts,
                             const ClangTidyCheck::OptionsView &Options) {
  if (Options.get("Skip", "false") == "true")
    return None;
  return tooling::makeRule(clang::ast_matchers::functionDecl(),
                           change(cat("void nothing()")), text("no message"));
}

class ConfigurableCheck : public TransformerClangTidyCheck {
public:
  ConfigurableCheck(StringRef Name, ClangTidyContext *Context)
      : TransformerClangTidyCheck(noSkip, Name, Context) {}
};

// Tests operation with config option "Skip" set to true and false.
TEST(TransformerClangTidyCheckTest, DisableByConfig) {
  const std::string Input = "void log(int);";
  const std::string Expected = "void nothing();";
  ClangTidyOptions Options;

  Options.CheckOptions["test-check-0.Skip"] = "true";
  EXPECT_EQ(Input, test::runCheckOnCode<ConfigurableCheck>(
                       Input, nullptr, "input.cc", None, Options));

  Options.CheckOptions["test-check-0.Skip"] = "false";
  EXPECT_EQ(Expected, test::runCheckOnCode<ConfigurableCheck>(
                          Input, nullptr, "input.cc", None, Options));
}

} // namespace
} // namespace utils
} // namespace tidy
} // namespace clang
