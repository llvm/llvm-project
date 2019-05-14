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
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace tooling;
using namespace ast_matchers;

namespace {
using ::testing::AllOf;
using ::testing::Eq;
using ::testing::HasSubstr;
using MatchResult = MatchFinder::MatchResult;
using tooling::stencil::node;
using tooling::stencil::sNode;
using tooling::stencil::text;

// In tests, we can't directly match on llvm::Expected since its accessors
// mutate the object. So, we collapse it to an Optional.
static llvm::Optional<std::string> toOptional(llvm::Expected<std::string> V) {
  if (V)
    return *V;
  ADD_FAILURE() << "Losing error in conversion to IsSomething: "
                << llvm::toString(V.takeError());
  return llvm::None;
}

// A very simple matcher for llvm::Optional values.
MATCHER_P(IsSomething, ValueMatcher, "") {
  if (!arg)
    return false;
  return ::testing::ExplainMatchResult(ValueMatcher, *arg, result_listener);
}

// Create a valid translation-unit from a statement.
static std::string wrapSnippet(llvm::Twine StatementCode) {
  return ("auto stencil_test_snippet = []{" + StatementCode + "};").str();
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
// matcher correspondingly. `Matcher` should match `StatementCode` exactly --
// that is, produce exactly one match.
static llvm::Optional<TestMatch> matchStmt(llvm::Twine StatementCode,
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
                                    [&Matcher](const llvm::StringError &Err) {
                                      EXPECT_THAT(Err.getMessage(), Matcher);
                                    });
      if (Err) {
        ADD_FAILURE() << "Unhandled error: " << llvm::toString(std::move(Err));
      }
    }
  }

  // Tests failures caused by references to unbound nodes. `unbound_id` is the
  // id that will cause the failure.
  void testUnboundNodeError(const Stencil &Stencil, llvm::StringRef UnboundId) {
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
  auto Stencil = Stencil::cat("if (!", node(Condition), ") ", sNode(Else),
                              " else ", sNode(Then));
  EXPECT_THAT(toOptional(Stencil.eval(StmtMatch->Result)),
              IsSomething(Eq("if (!true) return 0; else return 1;")));
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
  Stencil S = Stencil::cat("if (!", node(Condition), ") ", sNode(Else),
                              " else ", sNode(Then));
  EXPECT_THAT(toOptional(S(StmtMatch->Result)),
              IsSomething(Eq("if (!true) return 0; else return 1;")));
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
  auto Stencil = Stencil::cat("if(!", sNode("a1"), ") ", node("UNBOUND"), ";");
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
  EXPECT_THAT(toOptional(Stencil.eval(StmtMatch->Result)),
              IsSomething(Expected));
}

TEST_F(StencilTest, NodeOp) {
  StringRef Id = "id";
  testExpr(Id, "3;", Stencil::cat(node(Id)), "3");
}

TEST_F(StencilTest, SNodeOp) {
  StringRef Id = "id";
  testExpr(Id, "3;", Stencil::cat(sNode(Id)), "3;");
}

TEST(StencilEqualityTest, Equality) {
  using stencil::dPrint;
  auto Lhs = Stencil::cat("foo", node("node"), dPrint("dprint_id"));
  auto Rhs = Lhs;
  EXPECT_EQ(Lhs, Rhs);
}

TEST(StencilEqualityTest, InEqualityDifferentOrdering) {
  auto Lhs = Stencil::cat("foo", node("node"));
  auto Rhs = Stencil::cat(node("node"), "foo");
  EXPECT_NE(Lhs, Rhs);
}

TEST(StencilEqualityTest, InEqualityDifferentSizes) {
  auto Lhs = Stencil::cat("foo", node("node"), "bar", "baz");
  auto Rhs = Stencil::cat("foo", node("node"), "bar");
  EXPECT_NE(Lhs, Rhs);
}
} // namespace
