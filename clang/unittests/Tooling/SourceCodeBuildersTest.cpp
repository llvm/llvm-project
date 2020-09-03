//===- unittest/Tooling/SourceCodeBuildersTest.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Transformer/SourceCodeBuilders.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace tooling;
using namespace ast_matchers;

namespace {
using MatchResult = MatchFinder::MatchResult;
using llvm::ValueIs;

// Create a valid translation unit from a statement.
static std::string wrapSnippet(StringRef StatementCode) {
  return ("struct S { S(); S(int); int field; };\n"
          "S operator+(const S &a, const S &b);\n"
          "auto test_snippet = []{" +
          StatementCode + "};")
      .str();
}

static DeclarationMatcher wrapMatcher(const StatementMatcher &Matcher) {
  return varDecl(hasName("test_snippet"),
                 hasDescendant(compoundStmt(hasAnySubstatement(Matcher))));
}

struct TestMatch {
  // The AST unit from which `result` is built. We bundle it because it backs
  // the result. Users are not expected to access it.
  std::unique_ptr<ASTUnit> AstUnit;
  // The result to use in the test. References `ast_unit`.
  MatchResult Result;
};

// Matches `Matcher` against the statement `StatementCode` and returns the
// result. Handles putting the statement inside a function and modifying the
// matcher correspondingly. `Matcher` should match one of the statements in
// `StatementCode` exactly -- that is, produce exactly one match. However,
// `StatementCode` may contain other statements not described by `Matcher`.
static llvm::Optional<TestMatch> matchStmt(StringRef StatementCode,
                                           StatementMatcher Matcher) {
  auto AstUnit = buildASTFromCode(wrapSnippet(StatementCode));
  if (AstUnit == nullptr) {
    ADD_FAILURE() << "AST construction failed";
    return llvm::None;
  }
  ASTContext &Context = AstUnit->getASTContext();
  auto Matches = ast_matchers::match(wrapMatcher(Matcher), Context);
  // We expect a single, exact match for the statement.
  if (Matches.size() != 1) {
    ADD_FAILURE() << "Wrong number of matches: " << Matches.size();
    return llvm::None;
  }
  return TestMatch{std::move(AstUnit), MatchResult(Matches[0], &Context)};
}

static void testPredicate(bool (*Pred)(const Expr &), StringRef Snippet,
                          bool Expected) {
  auto StmtMatch = matchStmt(Snippet, expr().bind("expr"));
  ASSERT_TRUE(StmtMatch) << "Snippet: " << Snippet;
  EXPECT_EQ(Expected, Pred(*StmtMatch->Result.Nodes.getNodeAs<Expr>("expr")))
      << "Snippet: " << Snippet;
}

// Tests the predicate on the call argument, assuming `Snippet` is a function
// call.
static void testPredicateOnArg(bool (*Pred)(const Expr &), StringRef Snippet,
                               bool Expected) {
  auto StmtMatch = matchStmt(
      Snippet, expr(ignoringImplicit(callExpr(hasArgument(
                   0, ignoringElidableConstructorCall(expr().bind("arg")))))));
  ASSERT_TRUE(StmtMatch) << "Snippet: " << Snippet;
  EXPECT_EQ(Expected, Pred(*StmtMatch->Result.Nodes.getNodeAs<Expr>("arg")))
      << "Snippet: " << Snippet;
}

TEST(SourceCodeBuildersTest, needParensAfterUnaryOperator) {
  testPredicate(needParensAfterUnaryOperator, "3 + 5;", true);
  testPredicate(needParensAfterUnaryOperator, "true ? 3 : 5;", true);
  testPredicate(needParensAfterUnaryOperator, "S(3) + S(5);", true);

  testPredicate(needParensAfterUnaryOperator, "int x; x;", false);
  testPredicate(needParensAfterUnaryOperator, "int(3.0);", false);
  testPredicate(needParensAfterUnaryOperator, "void f(); f();", false);
  testPredicate(needParensAfterUnaryOperator, "int a[3]; a[0];", false);
  testPredicate(needParensAfterUnaryOperator, "S x; x.field;", false);
  testPredicate(needParensAfterUnaryOperator, "int x = 1; --x;", false);
  testPredicate(needParensAfterUnaryOperator, "int x = 1; -x;", false);
}

TEST(SourceCodeBuildersTest, needParensAfterUnaryOperatorInImplicitConversion) {
  // The binary operation will be embedded in various implicit
  // expressions. Verify they are ignored.
  testPredicateOnArg(needParensAfterUnaryOperator, "void f(S); f(3 + 5);",
                     true);
}

TEST(SourceCodeBuildersTest, mayEverNeedParens) {
  testPredicate(mayEverNeedParens, "3 + 5;", true);
  testPredicate(mayEverNeedParens, "true ? 3 : 5;", true);
  testPredicate(mayEverNeedParens, "int x = 1; --x;", true);
  testPredicate(mayEverNeedParens, "int x = 1; -x;", true);

  testPredicate(mayEverNeedParens, "int x; x;", false);
  testPredicate(mayEverNeedParens, "int(3.0);", false);
  testPredicate(mayEverNeedParens, "void f(); f();", false);
  testPredicate(mayEverNeedParens, "int a[3]; a[0];", false);
  testPredicate(mayEverNeedParens, "S x; x.field;", false);
}

TEST(SourceCodeBuildersTest, mayEverNeedParensInImplictConversion) {
  // The binary operation will be embedded in various implicit
  // expressions. Verify they are ignored.
  testPredicateOnArg(mayEverNeedParens, "void f(S); f(3 + 5);", true);
}

static void testBuilder(
    llvm::Optional<std::string> (*Builder)(const Expr &, const ASTContext &),
    StringRef Snippet, StringRef Expected) {
  auto StmtMatch = matchStmt(Snippet, expr().bind("expr"));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT(Builder(*StmtMatch->Result.Nodes.getNodeAs<Expr>("expr"),
                      *StmtMatch->Result.Context),
              ValueIs(std::string(Expected)));
}

TEST(SourceCodeBuildersTest, BuildParensUnaryOp) {
  testBuilder(buildParens, "-4;", "(-4)");
}

TEST(SourceCodeBuildersTest, BuildParensBinOp) {
  testBuilder(buildParens, "4 + 4;", "(4 + 4)");
}

TEST(SourceCodeBuildersTest, BuildParensValue) {
  testBuilder(buildParens, "4;", "4");
}

TEST(SourceCodeBuildersTest, BuildParensSubscript) {
  testBuilder(buildParens, "int a[3]; a[0];", "a[0]");
}

TEST(SourceCodeBuildersTest, BuildParensCall) {
  testBuilder(buildParens, "int f(int); f(4);", "f(4)");
}

TEST(SourceCodeBuildersTest, BuildAddressOfValue) {
  testBuilder(buildAddressOf, "S x; x;", "&x");
}

TEST(SourceCodeBuildersTest, BuildAddressOfPointerDereference) {
  testBuilder(buildAddressOf, "S *x; *x;", "x");
}

TEST(SourceCodeBuildersTest, BuildAddressOfPointerDereferenceIgnoresParens) {
  testBuilder(buildAddressOf, "S *x; *(x);", "x");
}

TEST(SourceCodeBuildersTest, BuildAddressOfBinaryOperation) {
  testBuilder(buildAddressOf, "S x; x + x;", "&(x + x)");
}

TEST(SourceCodeBuildersTest, BuildDereferencePointer) {
  testBuilder(buildDereference, "S *x; x;", "*x");
}

TEST(SourceCodeBuildersTest, BuildDereferenceValueAddress) {
  testBuilder(buildDereference, "S x; &x;", "x");
}

TEST(SourceCodeBuildersTest, BuildDereferenceValueAddressIgnoresParens) {
  testBuilder(buildDereference, "S x; &(x);", "x");
}

TEST(SourceCodeBuildersTest, BuildDereferenceBinaryOperation) {
  testBuilder(buildDereference, "S *x; x + 1;", "*(x + 1)");
}

TEST(SourceCodeBuildersTest, BuildDotValue) {
  testBuilder(buildDot, "S x; x;", "x.");
}

TEST(SourceCodeBuildersTest, BuildDotPointerDereference) {
  testBuilder(buildDot, "S *x; *x;", "x->");
}

TEST(SourceCodeBuildersTest, BuildDotPointerDereferenceIgnoresParens) {
  testBuilder(buildDot, "S *x; *(x);", "x->");
}

TEST(SourceCodeBuildersTest, BuildDotBinaryOperation) {
  testBuilder(buildDot, "S x; x + x;", "(x + x).");
}

TEST(SourceCodeBuildersTest, BuildDotPointerDereferenceExprWithParens) {
  testBuilder(buildDot, "S *x; *(x + 1);", "(x + 1)->");
}

TEST(SourceCodeBuildersTest, BuildArrowPointer) {
  testBuilder(buildArrow, "S *x; x;", "x->");
}

TEST(SourceCodeBuildersTest, BuildArrowValueAddress) {
  testBuilder(buildArrow, "S x; &x;", "x.");
}

TEST(SourceCodeBuildersTest, BuildArrowValueAddressIgnoresParens) {
  testBuilder(buildArrow, "S x; &(x);", "x.");
}

TEST(SourceCodeBuildersTest, BuildArrowBinaryOperation) {
  testBuilder(buildArrow, "S *x; x + 1;", "(x + 1)->");
}

TEST(SourceCodeBuildersTest, BuildArrowValueAddressWithParens) {
  testBuilder(buildArrow, "S x; &(true ? x : x);", "(true ? x : x).");
}
} // namespace
