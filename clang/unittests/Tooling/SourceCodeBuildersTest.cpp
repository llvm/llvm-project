//===- unittest/Tooling/SourceCodeBuildersTest.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Transformer/SourceCodeBuilders.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

using namespace clang;
using namespace tooling;
using namespace ast_matchers;

namespace {
using MatchResult = MatchFinder::MatchResult;
using llvm::ValueIs;

// Create a valid translation unit from a statement.
static std::string wrapSnippet(StringRef StatementCode) {
  return ("namespace std {\n"
          "template <typename T> struct unique_ptr {\n"
          "  T* operator->() const;\n"
          "  T& operator*() const;\n"
          "};\n"
          "template <typename T> struct shared_ptr {\n"
          "  T* operator->() const;\n"
          "  T& operator*() const;\n"
          "};\n"
          "}\n"
          "struct A { void super(); };\n"
          "struct S : public A { S(); S(int); int Field; };\n"
          "S operator+(const S &a, const S &b);\n"
          "struct Smart {\n"
          "  S* operator->() const;\n"
          "  S& operator*() const;\n"
          "};\n"
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
static std::optional<TestMatch> matchStmt(StringRef StatementCode,
                                          StatementMatcher Matcher) {
  auto AstUnit = buildASTFromCodeWithArgs(wrapSnippet(StatementCode),
                                          {"-Wno-unused-value"});
  if (AstUnit == nullptr) {
    ADD_FAILURE() << "AST construction failed";
    return std::nullopt;
  }
  ASTContext &Context = AstUnit->getASTContext();
  auto Matches = ast_matchers::match(wrapMatcher(Matcher), Context);
  // We expect a single, exact match for the statement.
  if (Matches.size() != 1) {
    ADD_FAILURE() << "Wrong number of matches: " << Matches.size();
    return std::nullopt;
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
  testPredicate(needParensAfterUnaryOperator, "S x; x.Field;", false);
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
  testPredicate(mayEverNeedParens, "S x; x.Field;", false);
}

TEST(SourceCodeBuildersTest, mayEverNeedParensInImplictConversion) {
  // The binary operation will be embedded in various implicit
  // expressions. Verify they are ignored.
  testPredicateOnArg(mayEverNeedParens, "void f(S); f(3 + 5);", true);
}

TEST(SourceCodeBuildersTest, isKnownPointerLikeTypeUniquePtr) {
  std::string Snippet = "std::unique_ptr<int> P; P;";
  auto StmtMatch =
      matchStmt(Snippet, declRefExpr(hasType(qualType().bind("ty"))));
  ASSERT_TRUE(StmtMatch) << "Snippet: " << Snippet;
  EXPECT_TRUE(
      isKnownPointerLikeType(*StmtMatch->Result.Nodes.getNodeAs<QualType>("ty"),
                             *StmtMatch->Result.Context))
      << "Snippet: " << Snippet;
}

TEST(SourceCodeBuildersTest, isKnownPointerLikeTypeSharedPtr) {
  std::string Snippet = "std::shared_ptr<int> P; P;";
  auto StmtMatch =
      matchStmt(Snippet, declRefExpr(hasType(qualType().bind("ty"))));
  ASSERT_TRUE(StmtMatch) << "Snippet: " << Snippet;
  EXPECT_TRUE(
      isKnownPointerLikeType(*StmtMatch->Result.Nodes.getNodeAs<QualType>("ty"),
                             *StmtMatch->Result.Context))
      << "Snippet: " << Snippet;
}

TEST(SourceCodeBuildersTest, isKnownPointerLikeTypeUnknownTypeFalse) {
  std::string Snippet = "Smart P; P;";
  auto StmtMatch =
      matchStmt(Snippet, declRefExpr(hasType(qualType().bind("ty"))));
  ASSERT_TRUE(StmtMatch) << "Snippet: " << Snippet;
  EXPECT_FALSE(
      isKnownPointerLikeType(*StmtMatch->Result.Nodes.getNodeAs<QualType>("ty"),
                             *StmtMatch->Result.Context))
      << "Snippet: " << Snippet;
}

TEST(SourceCodeBuildersTest, isKnownPointerLikeTypeNormalTypeFalse) {
  std::string Snippet = "int *P; P;";
  auto StmtMatch =
      matchStmt(Snippet, declRefExpr(hasType(qualType().bind("ty"))));
  ASSERT_TRUE(StmtMatch) << "Snippet: " << Snippet;
  EXPECT_FALSE(
      isKnownPointerLikeType(*StmtMatch->Result.Nodes.getNodeAs<QualType>("ty"),
                             *StmtMatch->Result.Context))
      << "Snippet: " << Snippet;
}

static void testBuilder(
    std::optional<std::string> (*Builder)(const Expr &, const ASTContext &),
    StringRef Snippet, StringRef Expected) {
  auto StmtMatch = matchStmt(Snippet, expr().bind("expr"));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT(Builder(*StmtMatch->Result.Nodes.getNodeAs<Expr>("expr"),
                      *StmtMatch->Result.Context),
              ValueIs(std::string(Expected)));
}

static void testBuildAccess(StringRef Snippet, StringRef Expected,
                            PLTClass C = PLTClass::Pointer) {
  auto StmtMatch = matchStmt(Snippet, expr().bind("expr"));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT(buildAccess(*StmtMatch->Result.Nodes.getNodeAs<Expr>("expr"),
                          *StmtMatch->Result.Context, C),
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

TEST(SourceCodeBuildersTest, BuildAddressOfImplicitThis) {
  StringRef Snippet = R"cc(
    struct Struct {
      void foo() {}
      void bar() {
        foo();
      }
    };
  )cc";
  auto StmtMatch = matchStmt(
      Snippet,
      cxxMemberCallExpr(onImplicitObjectArgument(cxxThisExpr().bind("expr"))));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT(buildAddressOf(*StmtMatch->Result.Nodes.getNodeAs<Expr>("expr"),
                             *StmtMatch->Result.Context),
              ValueIs(std::string("this")));
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

TEST(SourceCodeBuildersTest, BuildAccessValue) {
  testBuildAccess("S x; x;", "x.");
}

TEST(SourceCodeBuildersTest, BuildAccessPointerDereference) {
  testBuildAccess("S *x; *x;", "x->");
}

TEST(SourceCodeBuildersTest, BuildAccessPointerDereferenceIgnoresParens) {
  testBuildAccess("S *x; *(x);", "x->");
}

TEST(SourceCodeBuildersTest, BuildAccessValueBinaryOperation) {
  testBuildAccess("S x; x + x;", "(x + x).");
}

TEST(SourceCodeBuildersTest, BuildAccessPointerDereferenceExprWithParens) {
  testBuildAccess("S *x; *(x + 1);", "(x + 1)->");
}

TEST(SourceCodeBuildersTest, BuildAccessPointer) {
  testBuildAccess("S *x; x;", "x->");
}

TEST(SourceCodeBuildersTest, BuildAccessValueAddress) {
  testBuildAccess("S x; &x;", "x.");
}

TEST(SourceCodeBuildersTest, BuildAccessValueAddressIgnoresParens) {
  testBuildAccess("S x; &(x);", "x.");
}

TEST(SourceCodeBuildersTest, BuildAccessPointerBinaryOperation) {
  testBuildAccess("S *x; x + 1;", "(x + 1)->");
}

TEST(SourceCodeBuildersTest, BuildAccessValueAddressWithParens) {
  testBuildAccess("S x; &(true ? x : x);", "(true ? x : x).");
}

TEST(SourceCodeBuildersTest, BuildAccessSmartPointer) {
  testBuildAccess("std::unique_ptr<int> x; x;", "x->");
}

TEST(SourceCodeBuildersTest, BuildAccessSmartPointerAsValue) {
  testBuildAccess("std::unique_ptr<int> x; x;", "x.", PLTClass::Value);
}

TEST(SourceCodeBuildersTest, BuildAccessSmartPointerDeref) {
  testBuildAccess("std::unique_ptr<int> x; *x;", "x->");
}

TEST(SourceCodeBuildersTest, BuildAccessSmartPointerDerefAsValue) {
  testBuildAccess("std::unique_ptr<int> x; *x;", "(*x).", PLTClass::Value);
}

TEST(SourceCodeBuildersTest, BuildAccessSmartPointerMemberCall) {
  StringRef Snippet = R"cc(
    Smart x;
    x->Field;
  )cc";
  auto StmtMatch =
      matchStmt(Snippet, memberExpr(hasObjectExpression(expr().bind("expr"))));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT(buildAccess(*StmtMatch->Result.Nodes.getNodeAs<Expr>("expr"),
                          *StmtMatch->Result.Context),
              ValueIs(std::string("x->")));
}

TEST(SourceCodeBuildersTest, BuildAccessIgnoreImplicit) {
  StringRef Snippet = R"cc(
    S x;
    A *a;
    a = &x;
  )cc";
  auto StmtMatch =
      matchStmt(Snippet, binaryOperator(isAssignmentOperator(),
                                        hasRHS(expr().bind("expr"))));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT(buildAccess(*StmtMatch->Result.Nodes.getNodeAs<Expr>("expr"),
                          *StmtMatch->Result.Context),
              ValueIs(std::string("x.")));
}

TEST(SourceCodeBuildersTest, BuildAccessImplicitThis) {
  StringRef Snippet = R"cc(
    struct Struct {
      void foo() {}
      void bar() {
        foo();
      }
    };
  )cc";
  auto StmtMatch = matchStmt(
      Snippet,
      cxxMemberCallExpr(onImplicitObjectArgument(cxxThisExpr().bind("expr"))));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT(buildAccess(*StmtMatch->Result.Nodes.getNodeAs<Expr>("expr"),
                          *StmtMatch->Result.Context),
              ValueIs(std::string()));
}

TEST(SourceCodeBuildersTest, BuildAccessImplicitThisIgnoreImplicitCasts) {
  StringRef Snippet = "struct B : public A { void f() { super(); } };";
  auto StmtMatch = matchStmt(
      Snippet,
      cxxMemberCallExpr(onImplicitObjectArgument(expr().bind("expr"))));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT(buildAccess(*StmtMatch->Result.Nodes.getNodeAs<Expr>("expr"),
                          *StmtMatch->Result.Context),
              ValueIs(std::string()));
}
} // namespace
