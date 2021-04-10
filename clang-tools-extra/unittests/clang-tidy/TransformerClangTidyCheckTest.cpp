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
#include "clang/Tooling/Transformer/RangeSelector.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"
#include "clang/Tooling/Transformer/Transformer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace utils {
namespace {
using namespace ::clang::ast_matchers;

using transformer::cat;
using transformer::change;
using transformer::IncludeFormat;
using transformer::makeRule;
using transformer::node;
using transformer::noopEdit;
using transformer::RewriteRule;
using transformer::RootID;
using transformer::statement;

// Invert the code of an if-statement, while maintaining its semantics.
RewriteRule invertIf() {
  StringRef C = "C", T = "T", E = "E";
  RewriteRule Rule = makeRule(
      ifStmt(hasCondition(expr().bind(C)), hasThen(stmt().bind(T)),
             hasElse(stmt().bind(E))),
      change(statement(RootID), cat("if(!(", node(std::string(C)), ")) ",
                                    statement(std::string(E)), " else ",
                                    statement(std::string(T)))),
      cat("negate condition and reverse `then` and `else` branches"));
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

TEST(TransformerClangTidyCheckTest, DiagnosticsCorrectlyGenerated) {
  class DiagOnlyCheck : public TransformerClangTidyCheck {
  public:
    DiagOnlyCheck(StringRef Name, ClangTidyContext *Context)
        : TransformerClangTidyCheck(
              makeRule(returnStmt(), noopEdit(node(RootID)), cat("message")),
              Name, Context) {}
  };
  std::string Input = "int h() { return 5; }";
  std::vector<ClangTidyError> Errors;
  EXPECT_EQ(Input, test::runCheckOnCode<DiagOnlyCheck>(Input, &Errors));
  EXPECT_EQ(Errors.size(), 1U);
  EXPECT_EQ(Errors[0].Message.Message, "message");
  EXPECT_THAT(Errors[0].Message.Ranges, testing::IsEmpty());

  // The diagnostic is anchored to the match, "return 5".
  EXPECT_EQ(Errors[0].Message.FileOffset, 10U);
}

class IntLitCheck : public TransformerClangTidyCheck {
public:
  IntLitCheck(StringRef Name, ClangTidyContext *Context)
      : TransformerClangTidyCheck(
            makeRule(integerLiteral(), change(cat("LIT")), cat("no message")),
            Name, Context) {}
};

// Tests that two changes in a single macro expansion do not lead to conflicts
// in applying the changes.
TEST(TransformerClangTidyCheckTest, TwoChangesInOneMacroExpansion) {
  const std::string Input = R"cc(
#define PLUS(a,b) (a) + (b)
    int f() { return PLUS(3, 4); }
  )cc";
  const std::string Expected = R"cc(
#define PLUS(a,b) (a) + (b)
    int f() { return PLUS(LIT, LIT); }
  )cc";

  EXPECT_EQ(Expected, test::runCheckOnCode<IntLitCheck>(Input));
}

class BinOpCheck : public TransformerClangTidyCheck {
public:
  BinOpCheck(StringRef Name, ClangTidyContext *Context)
      : TransformerClangTidyCheck(
            makeRule(
                binaryOperator(hasOperatorName("+"), hasRHS(expr().bind("r"))),
                change(node("r"), cat("RIGHT")), cat("no message")),
            Name, Context) {}
};

// Tests case where the rule's match spans both source from the macro and its
// argument, while the change spans only the argument AND there are two such
// matches. We verify that both replacements succeed.
TEST(TransformerClangTidyCheckTest, TwoMatchesInMacroExpansion) {
  const std::string Input = R"cc(
#define M(a,b) (1 + a) * (1 + b)
    int f() { return M(3, 4); }
  )cc";
  const std::string Expected = R"cc(
#define M(a,b) (1 + a) * (1 + b)
    int f() { return M(RIGHT, RIGHT); }
  )cc";

  EXPECT_EQ(Expected, test::runCheckOnCode<BinOpCheck>(Input));
}

// A trivial rewrite-rule generator that requires Objective-C code.
Optional<RewriteRule> needsObjC(const LangOptions &LangOpts,
                                const ClangTidyCheck::OptionsView &Options) {
  if (!LangOpts.ObjC)
    return None;
  return makeRule(clang::ast_matchers::functionDecl(),
                  change(cat("void changed() {}")), cat("no message"));
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
  return makeRule(clang::ast_matchers::functionDecl(),
                  changeTo(cat("void nothing();")), cat("no message"));
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

RewriteRule replaceCall(IncludeFormat Format) {
  using namespace ::clang::ast_matchers;
  RewriteRule Rule = makeRule(callExpr(callee(functionDecl(hasName("f")))),
                              change(cat("other()")), cat("no message"));
  addInclude(Rule, "clang/OtherLib.h", Format);
  return Rule;
}

template <IncludeFormat Format>
class IncludeCheck : public TransformerClangTidyCheck {
public:
  IncludeCheck(StringRef Name, ClangTidyContext *Context)
      : TransformerClangTidyCheck(replaceCall(Format), Name, Context) {}
};

TEST(TransformerClangTidyCheckTest, AddIncludeQuoted) {

  std::string Input = R"cc(
    int f(int x);
    int h(int x) { return f(x); }
  )cc";
  std::string Expected = R"cc(#include "clang/OtherLib.h"


    int f(int x);
    int h(int x) { return other(); }
  )cc";

  EXPECT_EQ(Expected,
            test::runCheckOnCode<IncludeCheck<IncludeFormat::Quoted>>(Input));
}

TEST(TransformerClangTidyCheckTest, AddIncludeAngled) {
  std::string Input = R"cc(
    int f(int x);
    int h(int x) { return f(x); }
  )cc";
  std::string Expected = R"cc(#include <clang/OtherLib.h>


    int f(int x);
    int h(int x) { return other(); }
  )cc";

  EXPECT_EQ(Expected,
            test::runCheckOnCode<IncludeCheck<IncludeFormat::Angled>>(Input));
}

class IncludeOrderCheck : public TransformerClangTidyCheck {
  static RewriteRule rule() {
    using namespace ::clang::ast_matchers;
    RewriteRule Rule = transformer::makeRule(integerLiteral(), change(cat("5")),
                                             cat("no message"));
    addInclude(Rule, "bar.h", IncludeFormat::Quoted);
    return Rule;
  }

public:
  IncludeOrderCheck(StringRef Name, ClangTidyContext *Context)
      : TransformerClangTidyCheck(rule(), Name, Context) {}
};

TEST(TransformerClangTidyCheckTest, AddIncludeObeysSortStyleLocalOption) {
  std::string Input = R"cc(#include "input.h"
int h(int x) { return 3; })cc";

  std::string TreatsAsLibraryHeader = R"cc(#include "input.h"

#include "bar.h"
int h(int x) { return 5; })cc";

  std::string TreatsAsNormalHeader = R"cc(#include "bar.h"
#include "input.h"
int h(int x) { return 5; })cc";

  ClangTidyOptions Options;
  std::map<StringRef, StringRef> PathsToContent = {{"input.h", "\n"}};
  Options.CheckOptions["test-check-0.IncludeStyle"] = "llvm";
  EXPECT_EQ(TreatsAsLibraryHeader, test::runCheckOnCode<IncludeOrderCheck>(
                                       Input, nullptr, "inputTest.cpp", None,
                                       Options, PathsToContent));
  EXPECT_EQ(TreatsAsNormalHeader, test::runCheckOnCode<IncludeOrderCheck>(
                                      Input, nullptr, "input_test.cpp", None,
                                      Options, PathsToContent));

  Options.CheckOptions["test-check-0.IncludeStyle"] = "google";
  EXPECT_EQ(TreatsAsNormalHeader,
            test::runCheckOnCode<IncludeOrderCheck>(
                Input, nullptr, "inputTest.cc", None, Options, PathsToContent));
  EXPECT_EQ(TreatsAsLibraryHeader, test::runCheckOnCode<IncludeOrderCheck>(
                                       Input, nullptr, "input_test.cc", None,
                                       Options, PathsToContent));
}

TEST(TransformerClangTidyCheckTest, AddIncludeObeysSortStyleGlobalOption) {
  std::string Input = R"cc(#include "input.h"
int h(int x) { return 3; })cc";

  std::string TreatsAsLibraryHeader = R"cc(#include "input.h"

#include "bar.h"
int h(int x) { return 5; })cc";

  std::string TreatsAsNormalHeader = R"cc(#include "bar.h"
#include "input.h"
int h(int x) { return 5; })cc";

  ClangTidyOptions Options;
  std::map<StringRef, StringRef> PathsToContent = {{"input.h", "\n"}};
  Options.CheckOptions["IncludeStyle"] = "llvm";
  EXPECT_EQ(TreatsAsLibraryHeader, test::runCheckOnCode<IncludeOrderCheck>(
                                       Input, nullptr, "inputTest.cpp", None,
                                       Options, PathsToContent));
  EXPECT_EQ(TreatsAsNormalHeader, test::runCheckOnCode<IncludeOrderCheck>(
                                      Input, nullptr, "input_test.cpp", None,
                                      Options, PathsToContent));

  Options.CheckOptions["IncludeStyle"] = "google";
  EXPECT_EQ(TreatsAsNormalHeader,
            test::runCheckOnCode<IncludeOrderCheck>(
                Input, nullptr, "inputTest.cc", None, Options, PathsToContent));
  EXPECT_EQ(TreatsAsLibraryHeader, test::runCheckOnCode<IncludeOrderCheck>(
                                       Input, nullptr, "input_test.cc", None,
                                       Options, PathsToContent));
}

} // namespace
} // namespace utils
} // namespace tidy
} // namespace clang
