//===- unittest/Tooling/StencilTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/Stencil.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/FixIt.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace tooling;
using namespace ast_matchers;

namespace {
using ::llvm::Failed;
using ::llvm::HasValue;
using ::llvm::StringError;
using ::testing::AllOf;
using ::testing::Eq;
using ::testing::HasSubstr;
using MatchResult = MatchFinder::MatchResult;
using stencil::access;
using stencil::addressOf;
using stencil::cat;
using stencil::deref;
using stencil::dPrint;
using stencil::expression;
using stencil::ifBound;
using stencil::run;
using stencil::text;

// Create a valid translation-unit from a statement.
static std::string wrapSnippet(StringRef StatementCode) {
  return ("struct S { int field; }; auto stencil_test_snippet = []{" +
          StatementCode + "};")
      .str();
}

static DeclarationMatcher wrapMatcher(const StatementMatcher &Matcher) {
  return varDecl(hasName("stencil_test_snippet"),
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

class StencilTest : public ::testing::Test {
protected:
  // Verifies that the given stencil fails when evaluated on a valid match
  // result. Binds a statement to "stmt", a (non-member) ctor-initializer to
  // "init", an expression to "expr" and a (nameless) declaration to "decl".
  void testError(const Stencil &Stencil,
                 ::testing::Matcher<std::string> Matcher) {
    const std::string Snippet = R"cc(
      struct A {};
      class F : public A {
       public:
        F(int) {}
      };
      F(1);
    )cc";
    auto StmtMatch = matchStmt(
        Snippet,
        stmt(hasDescendant(
                 cxxConstructExpr(
                     hasDeclaration(decl(hasDescendant(cxxCtorInitializer(
                                                           isBaseInitializer())
                                                           .bind("init")))
                                        .bind("decl")))
                     .bind("expr")))
            .bind("stmt"));
    ASSERT_TRUE(StmtMatch);
    if (auto ResultOrErr = Stencil.eval(StmtMatch->Result)) {
      ADD_FAILURE() << "Expected failure but succeeded: " << *ResultOrErr;
    } else {
      auto Err = llvm::handleErrors(ResultOrErr.takeError(),
                                    [&Matcher](const StringError &Err) {
                                      EXPECT_THAT(Err.getMessage(), Matcher);
                                    });
      if (Err) {
        ADD_FAILURE() << "Unhandled error: " << llvm::toString(std::move(Err));
      }
    }
  }

  // Tests failures caused by references to unbound nodes. `unbound_id` is the
  // id that will cause the failure.
  void testUnboundNodeError(const Stencil &Stencil, StringRef UnboundId) {
    testError(Stencil, AllOf(HasSubstr(UnboundId), HasSubstr("not bound")));
  }
};

TEST_F(StencilTest, SingleStatement) {
  StringRef Condition("C"), Then("T"), Else("E");
  const std::string Snippet = R"cc(
    if (true)
      return 1;
    else
      return 0;
  )cc";
  auto StmtMatch = matchStmt(
      Snippet, ifStmt(hasCondition(expr().bind(Condition)),
                      hasThen(stmt().bind(Then)), hasElse(stmt().bind(Else))));
  ASSERT_TRUE(StmtMatch);
  // Invert the if-then-else.
  auto Stencil = cat("if (!", node(Condition), ") ", statement(Else), " else ",
                     statement(Then));
  EXPECT_THAT_EXPECTED(Stencil.eval(StmtMatch->Result),
                       HasValue("if (!true) return 0; else return 1;"));
}

TEST_F(StencilTest, SingleStatementCallOperator) {
  StringRef Condition("C"), Then("T"), Else("E");
  const std::string Snippet = R"cc(
    if (true)
      return 1;
    else
      return 0;
  )cc";
  auto StmtMatch = matchStmt(
      Snippet, ifStmt(hasCondition(expr().bind(Condition)),
                      hasThen(stmt().bind(Then)), hasElse(stmt().bind(Else))));
  ASSERT_TRUE(StmtMatch);
  // Invert the if-then-else.
  Stencil S = cat("if (!", node(Condition), ") ", statement(Else), " else ",
                  statement(Then));
  EXPECT_THAT_EXPECTED(S(StmtMatch->Result),
                       HasValue("if (!true) return 0; else return 1;"));
}

TEST_F(StencilTest, UnboundNode) {
  const std::string Snippet = R"cc(
    if (true)
      return 1;
    else
      return 0;
  )cc";
  auto StmtMatch = matchStmt(Snippet, ifStmt(hasCondition(stmt().bind("a1")),
                                             hasThen(stmt().bind("a2"))));
  ASSERT_TRUE(StmtMatch);
  auto Stencil = cat("if(!", node("a1"), ") ", node("UNBOUND"), ";");
  auto ResultOrErr = Stencil.eval(StmtMatch->Result);
  EXPECT_TRUE(llvm::errorToBool(ResultOrErr.takeError()))
      << "Expected unbound node, got " << *ResultOrErr;
}

// Tests that a stencil with a single parameter (`Id`) evaluates to the expected
// string, when `Id` is bound to the expression-statement in `Snippet`.
void testExpr(StringRef Id, StringRef Snippet, const Stencil &Stencil,
              StringRef Expected) {
  auto StmtMatch = matchStmt(Snippet, expr().bind(Id));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT_EXPECTED(Stencil.eval(StmtMatch->Result), HasValue(Expected));
}

void testFailure(StringRef Id, StringRef Snippet, const Stencil &Stencil,
                 testing::Matcher<std::string> MessageMatcher) {
  auto StmtMatch = matchStmt(Snippet, expr().bind(Id));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT_EXPECTED(Stencil.eval(StmtMatch->Result),
                       Failed<StringError>(testing::Property(
                           &StringError::getMessage, MessageMatcher)));
}

TEST_F(StencilTest, SelectionOp) {
  StringRef Id = "id";
  testExpr(Id, "3;", cat(node(Id)), "3");
}

TEST_F(StencilTest, IfBoundOpBound) {
  StringRef Id = "id";
  testExpr(Id, "3;", cat(ifBound(Id, text("5"), text("7"))), "5");
}

TEST_F(StencilTest, IfBoundOpUnbound) {
  StringRef Id = "id";
  testExpr(Id, "3;", cat(ifBound("other", text("5"), text("7"))), "7");
}

TEST_F(StencilTest, ExpressionOpNoParens) {
  StringRef Id = "id";
  testExpr(Id, "3;", cat(expression(Id)), "3");
}

// Don't parenthesize a parens expression.
TEST_F(StencilTest, ExpressionOpNoParensParens) {
  StringRef Id = "id";
  testExpr(Id, "(3);", cat(expression(Id)), "(3)");
}

TEST_F(StencilTest, ExpressionOpBinaryOpParens) {
  StringRef Id = "id";
  testExpr(Id, "3+4;", cat(expression(Id)), "(3+4)");
}

// `expression` shares code with other ops, so we get sufficient coverage of the
// error handling code with this test. If that changes in the future, more error
// tests should be added.
TEST_F(StencilTest, ExpressionOpUnbound) {
  StringRef Id = "id";
  testFailure(Id, "3;", cat(expression("ACACA")),
              AllOf(HasSubstr("ACACA"), HasSubstr("not bound")));
}

TEST_F(StencilTest, DerefPointer) {
  StringRef Id = "id";
  testExpr(Id, "int *x; x;", cat(deref(Id)), "*x");
}

TEST_F(StencilTest, DerefBinOp) {
  StringRef Id = "id";
  testExpr(Id, "int *x; x + 1;", cat(deref(Id)), "*(x + 1)");
}

TEST_F(StencilTest, DerefAddressExpr) {
  StringRef Id = "id";
  testExpr(Id, "int x; &x;", cat(deref(Id)), "x");
}

TEST_F(StencilTest, AddressOfValue) {
  StringRef Id = "id";
  testExpr(Id, "int x; x;", cat(addressOf(Id)), "&x");
}

TEST_F(StencilTest, AddressOfDerefExpr) {
  StringRef Id = "id";
  testExpr(Id, "int *x; *x;", cat(addressOf(Id)), "x");
}

TEST_F(StencilTest, AccessOpValue) {
  StringRef Snippet = R"cc(
    S x;
    x;
  )cc";
  StringRef Id = "id";
  testExpr(Id, Snippet, cat(access(Id, "field")), "x.field");
}

TEST_F(StencilTest, AccessOpValueExplicitText) {
  StringRef Snippet = R"cc(
    S x;
    x;
  )cc";
  StringRef Id = "id";
  testExpr(Id, Snippet, cat(access(Id, text("field"))), "x.field");
}

TEST_F(StencilTest, AccessOpValueAddress) {
  StringRef Snippet = R"cc(
    S x;
    &x;
  )cc";
  StringRef Id = "id";
  testExpr(Id, Snippet, cat(access(Id, "field")), "x.field");
}

TEST_F(StencilTest, AccessOpPointer) {
  StringRef Snippet = R"cc(
    S *x;
    x;
  )cc";
  StringRef Id = "id";
  testExpr(Id, Snippet, cat(access(Id, "field")), "x->field");
}

TEST_F(StencilTest, AccessOpPointerDereference) {
  StringRef Snippet = R"cc(
    S *x;
    *x;
  )cc";
  StringRef Id = "id";
  testExpr(Id, Snippet, cat(access(Id, "field")), "x->field");
}

TEST_F(StencilTest, AccessOpExplicitThis) {
  using clang::ast_matchers::hasObjectExpression;
  using clang::ast_matchers::memberExpr;

  // Set up the code so we can bind to a use of this.
  StringRef Snippet = R"cc(
    class C {
     public:
      int x;
      int foo() { return this->x; }
    };
  )cc";
  auto StmtMatch =
      matchStmt(Snippet, returnStmt(hasReturnValue(ignoringImplicit(memberExpr(
                             hasObjectExpression(expr().bind("obj")))))));
  ASSERT_TRUE(StmtMatch);
  const Stencil Stencil = cat(access("obj", "field"));
  EXPECT_THAT_EXPECTED(Stencil.eval(StmtMatch->Result),
                       HasValue("this->field"));
}

TEST_F(StencilTest, AccessOpImplicitThis) {
  using clang::ast_matchers::hasObjectExpression;
  using clang::ast_matchers::memberExpr;

  // Set up the code so we can bind to a use of (implicit) this.
  StringRef Snippet = R"cc(
    class C {
     public:
      int x;
      int foo() { return x; }
    };
  )cc";
  auto StmtMatch =
      matchStmt(Snippet, returnStmt(hasReturnValue(ignoringImplicit(memberExpr(
                             hasObjectExpression(expr().bind("obj")))))));
  ASSERT_TRUE(StmtMatch);
  const Stencil Stencil = cat(access("obj", "field"));
  EXPECT_THAT_EXPECTED(Stencil.eval(StmtMatch->Result), HasValue("field"));
}

TEST_F(StencilTest, RunOp) {
  StringRef Id = "id";
  auto SimpleFn = [Id](const MatchResult &R) {
    return std::string(R.Nodes.getNodeAs<Stmt>(Id) != nullptr ? "Bound"
                                                              : "Unbound");
  };
  testExpr(Id, "3;", cat(run(SimpleFn)), "Bound");
}

TEST(StencilEqualityTest, Equality) {
  auto Lhs = cat("foo", dPrint("dprint_id"));
  auto Rhs = cat("foo", dPrint("dprint_id"));
  EXPECT_EQ(Lhs, Rhs);
}

TEST(StencilEqualityTest, InEqualityDifferentOrdering) {
  auto Lhs = cat("foo", dPrint("node"));
  auto Rhs = cat(dPrint("node"), "foo");
  EXPECT_NE(Lhs, Rhs);
}

TEST(StencilEqualityTest, InEqualityDifferentSizes) {
  auto Lhs = cat("foo", dPrint("node"), "bar", "baz");
  auto Rhs = cat("foo", dPrint("node"), "bar");
  EXPECT_NE(Lhs, Rhs);
}

// node is opaque and therefore cannot be examined for equality.
TEST(StencilEqualityTest, InEqualitySelection) {
  auto S1 = cat(node("node"));
  auto S2 = cat(node("node"));
  EXPECT_NE(S1, S2);
}

// `run` is opaque.
TEST(StencilEqualityTest, InEqualityRun) {
  auto F = [](const MatchResult &R) { return "foo"; };
  auto S1 = cat(run(F));
  auto S2 = cat(run(F));
  EXPECT_NE(S1, S2);
}

TEST(StencilToStringTest, RawTextOp) {
  auto S = cat("foo bar baz");
  StringRef Expected = R"("foo bar baz")";
  EXPECT_EQ(S.toString(), Expected);
}

TEST(StencilToStringTest, RawTextOpEscaping) {
  auto S = cat("foo \"bar\" baz\\n");
  StringRef Expected = R"("foo \"bar\" baz\\n")";
  EXPECT_EQ(S.toString(), Expected);
}

TEST(StencilToStringTest, DebugPrintNodeOp) {
  auto S = cat(dPrint("Id"));
  StringRef Expected = R"repr(dPrint("Id"))repr";
  EXPECT_EQ(S.toString(), Expected);
}

TEST(StencilToStringTest, ExpressionOp) {
  auto S = cat(expression("Id"));
  StringRef Expected = R"repr(expression("Id"))repr";
  EXPECT_EQ(S.toString(), Expected);
}

TEST(StencilToStringTest, DerefOp) {
  auto S = cat(deref("Id"));
  StringRef Expected = R"repr(deref("Id"))repr";
  EXPECT_EQ(S.toString(), Expected);
}

TEST(StencilToStringTest, AddressOfOp) {
  auto S = cat(addressOf("Id"));
  StringRef Expected = R"repr(addressOf("Id"))repr";
  EXPECT_EQ(S.toString(), Expected);
}

TEST(StencilToStringTest, AccessOp) {
  auto S = cat(access("Id", text("memberData")));
  StringRef Expected = R"repr(access("Id", "memberData"))repr";
  EXPECT_EQ(S.toString(), Expected);
}

TEST(StencilToStringTest, AccessOpStencilPart) {
  auto S = cat(access("Id", access("subId", "memberData")));
  StringRef Expected = R"repr(access("Id", access("subId", "memberData")))repr";
  EXPECT_EQ(S.toString(), Expected);
}

TEST(StencilToStringTest, IfBoundOp) {
  auto S = cat(ifBound("Id", text("trueText"), access("exprId", "memberData")));
  StringRef Expected =
      R"repr(ifBound("Id", "trueText", access("exprId", "memberData")))repr";
  EXPECT_EQ(S.toString(), Expected);
}

TEST(StencilToStringTest, MultipleOp) {
  auto S = cat("foo", access("x", "m()"), "bar",
               ifBound("x", text("t"), access("e", "f")));
  StringRef Expected = R"repr("foo", access("x", "m()"), "bar", )repr"
                       R"repr(ifBound("x", "t", access("e", "f")))repr";
  EXPECT_EQ(S.toString(), Expected);
}
} // namespace
