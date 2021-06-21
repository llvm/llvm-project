//===- unittest/Tooling/StencilTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Transformer/Stencil.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/FixIt.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace transformer;
using namespace ast_matchers;

namespace {
using ::llvm::Failed;
using ::llvm::HasValue;
using ::llvm::StringError;
using ::testing::AllOf;
using ::testing::HasSubstr;
using MatchResult = MatchFinder::MatchResult;

// Create a valid translation-unit from a statement.
static std::string wrapSnippet(StringRef ExtraPreface,
                               StringRef StatementCode) {
  constexpr char Preface[] = R"cc(
    namespace N { class C {}; }
    namespace { class AnonC {}; }
    struct S { int Field; };
    struct Smart {
      S* operator->() const;
      S& operator*() const;
    };
  )cc";
  return (Preface + ExtraPreface + "auto stencil_test_snippet = []{" +
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
// `ExtraPreface` (optionally) adds extra decls to the TU, before the code.
static llvm::Optional<TestMatch> matchStmt(StringRef StatementCode,
                                           StatementMatcher Matcher,
                                           StringRef ExtraPreface = "") {
  auto AstUnit = tooling::buildASTFromCodeWithArgs(
      wrapSnippet(ExtraPreface, StatementCode), {"-Wno-unused-value"});
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
    if (auto ResultOrErr = Stencil->eval(StmtMatch->Result)) {
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
    testError(Stencil,
              AllOf(HasSubstr(std::string(UnboundId)), HasSubstr("not bound")));
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
  auto Stencil =
      cat("if (!", node(std::string(Condition)), ") ",
          statement(std::string(Else)), " else ", statement(std::string(Then)));
  EXPECT_THAT_EXPECTED(Stencil->eval(StmtMatch->Result),
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
  auto ResultOrErr = Stencil->eval(StmtMatch->Result);
  EXPECT_TRUE(llvm::errorToBool(ResultOrErr.takeError()))
      << "Expected unbound node, got " << *ResultOrErr;
}

// Tests that a stencil with a single parameter (`Id`) evaluates to the expected
// string, when `Id` is bound to the expression-statement in `Snippet`.
void testExpr(StringRef Id, StringRef Snippet, const Stencil &Stencil,
              StringRef Expected) {
  auto StmtMatch = matchStmt(Snippet, expr().bind(Id));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT_EXPECTED(Stencil->eval(StmtMatch->Result),
                       HasValue(std::string(Expected)));
}

void testFailure(StringRef Id, StringRef Snippet, const Stencil &Stencil,
                 testing::Matcher<std::string> MessageMatcher) {
  auto StmtMatch = matchStmt(Snippet, expr().bind(Id));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT_EXPECTED(Stencil->eval(StmtMatch->Result),
                       Failed<StringError>(testing::Property(
                           &StringError::getMessage, MessageMatcher)));
}

TEST_F(StencilTest, SelectionOp) {
  StringRef Id = "id";
  testExpr(Id, "3;", cat(node(std::string(Id))), "3");
}

TEST_F(StencilTest, IfBoundOpBound) {
  StringRef Id = "id";
  testExpr(Id, "3;", ifBound(Id, cat("5"), cat("7")), "5");
}

TEST_F(StencilTest, IfBoundOpUnbound) {
  StringRef Id = "id";
  testExpr(Id, "3;", ifBound("other", cat("5"), cat("7")), "7");
}

TEST_F(StencilTest, ExpressionOpNoParens) {
  StringRef Id = "id";
  testExpr(Id, "3;", expression(Id), "3");
}

// Don't parenthesize a parens expression.
TEST_F(StencilTest, ExpressionOpNoParensParens) {
  StringRef Id = "id";
  testExpr(Id, "(3);", expression(Id), "(3)");
}

TEST_F(StencilTest, ExpressionOpBinaryOpParens) {
  StringRef Id = "id";
  testExpr(Id, "3+4;", expression(Id), "(3+4)");
}

// `expression` shares code with other ops, so we get sufficient coverage of the
// error handling code with this test. If that changes in the future, more error
// tests should be added.
TEST_F(StencilTest, ExpressionOpUnbound) {
  StringRef Id = "id";
  testFailure(Id, "3;", expression("ACACA"),
              AllOf(HasSubstr("ACACA"), HasSubstr("not bound")));
}

TEST_F(StencilTest, DerefPointer) {
  StringRef Id = "id";
  testExpr(Id, "int *x; x;", deref(Id), "*x");
}

TEST_F(StencilTest, DerefBinOp) {
  StringRef Id = "id";
  testExpr(Id, "int *x; x + 1;", deref(Id), "*(x + 1)");
}

TEST_F(StencilTest, DerefAddressExpr) {
  StringRef Id = "id";
  testExpr(Id, "int x; &x;", deref(Id), "x");
}

TEST_F(StencilTest, AddressOfValue) {
  StringRef Id = "id";
  testExpr(Id, "int x; x;", addressOf(Id), "&x");
}

TEST_F(StencilTest, AddressOfDerefExpr) {
  StringRef Id = "id";
  testExpr(Id, "int *x; *x;", addressOf(Id), "x");
}

TEST_F(StencilTest, MaybeDerefValue) {
  StringRef Id = "id";
  testExpr(Id, "int x; x;", maybeDeref(Id), "x");
}

TEST_F(StencilTest, MaybeDerefPointer) {
  StringRef Id = "id";
  testExpr(Id, "int *x; x;", maybeDeref(Id), "*x");
}

TEST_F(StencilTest, MaybeDerefBinOp) {
  StringRef Id = "id";
  testExpr(Id, "int *x; x + 1;", maybeDeref(Id), "*(x + 1)");
}

TEST_F(StencilTest, MaybeDerefAddressExpr) {
  StringRef Id = "id";
  testExpr(Id, "int x; &x;", maybeDeref(Id), "x");
}

TEST_F(StencilTest, MaybeDerefSmartPointer) {
  StringRef Id = "id";
  std::string Snippet = R"cc(
    Smart x;
    x;
  )cc";
  testExpr(Id, Snippet, maybeDeref(Id), "*x");
}

// Tests that unique_ptr specifically is handled.
TEST_F(StencilTest, MaybeDerefSmartPointerUniquePtr) {
  StringRef Id = "id";
  // We deliberately specify `unique_ptr` as empty to verify that it matches
  // because of its name, rather than its contents.
  StringRef ExtraPreface =
      "namespace std { template <typename T> class unique_ptr {}; }\n";
  StringRef Snippet = R"cc(
    std::unique_ptr<int> x;
    x;
  )cc";
  auto StmtMatch = matchStmt(Snippet, expr().bind(Id), ExtraPreface);
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT_EXPECTED(maybeDeref(Id)->eval(StmtMatch->Result),
                       HasValue(std::string("*x")));
}

TEST_F(StencilTest, MaybeDerefSmartPointerFromMemberExpr) {
  StringRef Id = "id";
  std::string Snippet = "Smart x; x->Field;";
  auto StmtMatch =
      matchStmt(Snippet, memberExpr(hasObjectExpression(expr().bind(Id))));
  ASSERT_TRUE(StmtMatch);
  const Stencil Stencil = maybeDeref(Id);
  EXPECT_THAT_EXPECTED(Stencil->eval(StmtMatch->Result), HasValue("*x"));
}

TEST_F(StencilTest, MaybeAddressOfPointer) {
  StringRef Id = "id";
  testExpr(Id, "int *x; x;", maybeAddressOf(Id), "x");
}

TEST_F(StencilTest, MaybeAddressOfValue) {
  StringRef Id = "id";
  testExpr(Id, "int x; x;", addressOf(Id), "&x");
}

TEST_F(StencilTest, MaybeAddressOfBinOp) {
  StringRef Id = "id";
  testExpr(Id, "int x; x + 1;", maybeAddressOf(Id), "&(x + 1)");
}

TEST_F(StencilTest, MaybeAddressOfDerefExpr) {
  StringRef Id = "id";
  testExpr(Id, "int *x; *x;", addressOf(Id), "x");
}

TEST_F(StencilTest, MaybeAddressOfSmartPointer) {
  StringRef Id = "id";
  testExpr(Id, "Smart x; x;", maybeAddressOf(Id), "x");
}

TEST_F(StencilTest, MaybeAddressOfSmartPointerFromMemberCall) {
  StringRef Id = "id";
  std::string Snippet = "Smart x; x->Field;";
  auto StmtMatch =
      matchStmt(Snippet, memberExpr(hasObjectExpression(expr().bind(Id))));
  ASSERT_TRUE(StmtMatch);
  const Stencil Stencil = maybeAddressOf(Id);
  EXPECT_THAT_EXPECTED(Stencil->eval(StmtMatch->Result), HasValue("x"));
}

TEST_F(StencilTest, MaybeAddressOfSmartPointerDerefNoCancel) {
  StringRef Id = "id";
  testExpr(Id, "Smart x; *x;", maybeAddressOf(Id), "&*x");
}

TEST_F(StencilTest, AccessOpValue) {
  StringRef Snippet = R"cc(
    S x;
    x;
  )cc";
  StringRef Id = "id";
  testExpr(Id, Snippet, access(Id, "field"), "x.field");
}

TEST_F(StencilTest, AccessOpValueExplicitText) {
  StringRef Snippet = R"cc(
    S x;
    x;
  )cc";
  StringRef Id = "id";
  testExpr(Id, Snippet, access(Id, cat("field")), "x.field");
}

TEST_F(StencilTest, AccessOpValueAddress) {
  StringRef Snippet = R"cc(
    S x;
    &x;
  )cc";
  StringRef Id = "id";
  testExpr(Id, Snippet, access(Id, "field"), "x.field");
}

TEST_F(StencilTest, AccessOpPointer) {
  StringRef Snippet = R"cc(
    S *x;
    x;
  )cc";
  StringRef Id = "id";
  testExpr(Id, Snippet, access(Id, "field"), "x->field");
}

TEST_F(StencilTest, AccessOpPointerDereference) {
  StringRef Snippet = R"cc(
    S *x;
    *x;
  )cc";
  StringRef Id = "id";
  testExpr(Id, Snippet, access(Id, "field"), "x->field");
}

TEST_F(StencilTest, AccessOpSmartPointer) {
  StringRef Snippet = R"cc(
    Smart x;
    x;
  )cc";
  StringRef Id = "id";
  testExpr(Id, Snippet, access(Id, "field"), "x->field");
}

TEST_F(StencilTest, AccessOpSmartPointerDereference) {
  StringRef Snippet = R"cc(
    Smart x;
    *x;
  )cc";
  StringRef Id = "id";
  testExpr(Id, Snippet, access(Id, "field"), "x->field");
}

TEST_F(StencilTest, AccessOpSmartPointerMemberCall) {
  StringRef Snippet = R"cc(
    Smart x;
    x->Field;
  )cc";
  StringRef Id = "id";
  auto StmtMatch =
      matchStmt(Snippet, memberExpr(hasObjectExpression(expr().bind(Id))));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT_EXPECTED(access(Id, "field")->eval(StmtMatch->Result),
                       HasValue("x->field"));
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
  auto StmtMatch = matchStmt(
      Snippet,
      traverse(TK_AsIs, returnStmt(hasReturnValue(ignoringImplicit(memberExpr(
                            hasObjectExpression(expr().bind("obj"))))))));
  ASSERT_TRUE(StmtMatch);
  const Stencil Stencil = access("obj", "field");
  EXPECT_THAT_EXPECTED(Stencil->eval(StmtMatch->Result),
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
  const Stencil Stencil = access("obj", "field");
  EXPECT_THAT_EXPECTED(Stencil->eval(StmtMatch->Result), HasValue("field"));
}

TEST_F(StencilTest, DescribeType) {
  std::string Snippet = "int *x; x;";
  std::string Expected = "int *";
  auto StmtMatch =
      matchStmt(Snippet, declRefExpr(hasType(qualType().bind("type"))));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT_EXPECTED(describe("type")->eval(StmtMatch->Result),
                       HasValue(std::string(Expected)));
}

TEST_F(StencilTest, DescribeSugaredType) {
  std::string Snippet = "using Ty = int; Ty *x; x;";
  std::string Expected = "Ty *";
  auto StmtMatch =
      matchStmt(Snippet, declRefExpr(hasType(qualType().bind("type"))));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT_EXPECTED(describe("type")->eval(StmtMatch->Result),
                       HasValue(std::string(Expected)));
}

TEST_F(StencilTest, DescribeDeclType) {
  std::string Snippet = "S s; s;";
  std::string Expected = "S";
  auto StmtMatch =
      matchStmt(Snippet, declRefExpr(hasType(qualType().bind("type"))));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT_EXPECTED(describe("type")->eval(StmtMatch->Result),
                       HasValue(std::string(Expected)));
}

TEST_F(StencilTest, DescribeQualifiedType) {
  std::string Snippet = "N::C c; c;";
  std::string Expected = "N::C";
  auto StmtMatch =
      matchStmt(Snippet, declRefExpr(hasType(qualType().bind("type"))));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT_EXPECTED(describe("type")->eval(StmtMatch->Result),
                       HasValue(std::string(Expected)));
}

TEST_F(StencilTest, DescribeUnqualifiedType) {
  std::string Snippet = "using N::C; C c; c;";
  std::string Expected = "N::C";
  auto StmtMatch =
      matchStmt(Snippet, declRefExpr(hasType(qualType().bind("type"))));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT_EXPECTED(describe("type")->eval(StmtMatch->Result),
                       HasValue(std::string(Expected)));
}

TEST_F(StencilTest, DescribeAnonNamespaceType) {
  std::string Snippet = "AnonC c; c;";
  std::string Expected = "(anonymous namespace)::AnonC";
  auto StmtMatch =
      matchStmt(Snippet, declRefExpr(hasType(qualType().bind("type"))));
  ASSERT_TRUE(StmtMatch);
  EXPECT_THAT_EXPECTED(describe("type")->eval(StmtMatch->Result),
                       HasValue(std::string(Expected)));
}

TEST_F(StencilTest, RunOp) {
  StringRef Id = "id";
  auto SimpleFn = [Id](const MatchResult &R) {
    return std::string(R.Nodes.getNodeAs<Stmt>(Id) != nullptr ? "Bound"
                                                              : "Unbound");
  };
  testExpr(Id, "3;", run(SimpleFn), "Bound");
}

TEST_F(StencilTest, CatOfMacroRangeSucceeds) {
  StringRef Snippet = R"cpp(
#define MACRO 3.77
  double foo(double d);
  foo(MACRO);)cpp";

  auto StmtMatch =
      matchStmt(Snippet, callExpr(callee(functionDecl(hasName("foo"))),
                                  argumentCountIs(1),
                                  hasArgument(0, expr().bind("arg"))));
  ASSERT_TRUE(StmtMatch);
  Stencil S = cat(node("arg"));
  EXPECT_THAT_EXPECTED(S->eval(StmtMatch->Result), HasValue("MACRO"));
}

TEST_F(StencilTest, CatOfMacroArgRangeSucceeds) {
  StringRef Snippet = R"cpp(
#define MACRO(a, b) a + b
  MACRO(2, 3);)cpp";

  auto StmtMatch =
      matchStmt(Snippet, binaryOperator(hasRHS(expr().bind("rhs"))));
  ASSERT_TRUE(StmtMatch);
  Stencil S = cat(node("rhs"));
  EXPECT_THAT_EXPECTED(S->eval(StmtMatch->Result), HasValue("3"));
}

TEST_F(StencilTest, CatOfMacroArgSubRangeSucceeds) {
  StringRef Snippet = R"cpp(
#define MACRO(a, b) a + b
  int foo(int);
  MACRO(2, foo(3));)cpp";

  auto StmtMatch = matchStmt(
      Snippet, binaryOperator(hasRHS(callExpr(
                   callee(functionDecl(hasName("foo"))), argumentCountIs(1),
                   hasArgument(0, expr().bind("arg"))))));
  ASSERT_TRUE(StmtMatch);
  Stencil S = cat(node("arg"));
  EXPECT_THAT_EXPECTED(S->eval(StmtMatch->Result), HasValue("3"));
}

TEST_F(StencilTest, CatOfInvalidRangeFails) {
  StringRef Snippet = R"cpp(
#define MACRO (3.77)
  double foo(double d);
  foo(MACRO);)cpp";

  auto StmtMatch =
      matchStmt(Snippet, callExpr(callee(functionDecl(hasName("foo"))),
                                  argumentCountIs(1),
                                  hasArgument(0, expr().bind("arg"))));
  ASSERT_TRUE(StmtMatch);
  Stencil S = cat(node("arg"));
  Expected<std::string> Result = S->eval(StmtMatch->Result);
  ASSERT_THAT_EXPECTED(Result, Failed<StringError>());
  llvm::handleAllErrors(Result.takeError(), [](const llvm::StringError &E) {
    EXPECT_THAT(E.getMessage(), AllOf(HasSubstr("selected range"),
                                      HasSubstr("macro expansion")));
  });
}

// The `StencilToStringTest` tests verify that the string representation of the
// stencil combinator matches (as best possible) the spelling of the
// combinator's construction.  Exceptions include those combinators that have no
// explicit spelling (like raw text) and those supporting non-printable
// arguments (like `run`, `selection`).

TEST(StencilToStringTest, RawTextOp) {
  auto S = cat("foo bar baz");
  StringRef Expected = R"("foo bar baz")";
  EXPECT_EQ(S->toString(), Expected);
}

TEST(StencilToStringTest, RawTextOpEscaping) {
  auto S = cat("foo \"bar\" baz\\n");
  StringRef Expected = R"("foo \"bar\" baz\\n")";
  EXPECT_EQ(S->toString(), Expected);
}

TEST(StencilToStringTest, DescribeOp) {
  auto S = describe("Id");
  StringRef Expected = R"repr(describe("Id"))repr";
  EXPECT_EQ(S->toString(), Expected);
}

TEST(StencilToStringTest, DebugPrintNodeOp) {
  auto S = dPrint("Id");
  StringRef Expected = R"repr(dPrint("Id"))repr";
  EXPECT_EQ(S->toString(), Expected);
}

TEST(StencilToStringTest, ExpressionOp) {
  auto S = expression("Id");
  StringRef Expected = R"repr(expression("Id"))repr";
  EXPECT_EQ(S->toString(), Expected);
}

TEST(StencilToStringTest, DerefOp) {
  auto S = deref("Id");
  StringRef Expected = R"repr(deref("Id"))repr";
  EXPECT_EQ(S->toString(), Expected);
}

TEST(StencilToStringTest, AddressOfOp) {
  auto S = addressOf("Id");
  StringRef Expected = R"repr(addressOf("Id"))repr";
  EXPECT_EQ(S->toString(), Expected);
}

TEST(StencilToStringTest, SelectionOp) {
  auto S1 = cat(node("node1"));
  EXPECT_EQ(S1->toString(), "selection(...)");
}

TEST(StencilToStringTest, AccessOpText) {
  auto S = access("Id", "memberData");
  StringRef Expected = R"repr(access("Id", "memberData"))repr";
  EXPECT_EQ(S->toString(), Expected);
}

TEST(StencilToStringTest, AccessOpSelector) {
  auto S = access("Id", cat(name("otherId")));
  StringRef Expected = R"repr(access("Id", selection(...)))repr";
  EXPECT_EQ(S->toString(), Expected);
}

TEST(StencilToStringTest, AccessOpStencil) {
  auto S = access("Id", cat("foo_", "bar"));
  StringRef Expected = R"repr(access("Id", seq("foo_", "bar")))repr";
  EXPECT_EQ(S->toString(), Expected);
}

TEST(StencilToStringTest, IfBoundOp) {
  auto S = ifBound("Id", cat("trueText"), access("exprId", "memberData"));
  StringRef Expected =
      R"repr(ifBound("Id", "trueText", access("exprId", "memberData")))repr";
  EXPECT_EQ(S->toString(), Expected);
}

TEST(StencilToStringTest, RunOp) {
  auto F1 = [](const MatchResult &R) { return "foo"; };
  auto S1 = run(F1);
  EXPECT_EQ(S1->toString(), "run(...)");
}

TEST(StencilToStringTest, Sequence) {
  auto S = cat("foo", access("x", "m()"), "bar",
               ifBound("x", cat("t"), access("e", "f")));
  StringRef Expected = R"repr(seq("foo", access("x", "m()"), "bar", )repr"
                       R"repr(ifBound("x", "t", access("e", "f"))))repr";
  EXPECT_EQ(S->toString(), Expected);
}

TEST(StencilToStringTest, SequenceEmpty) {
  auto S = cat();
  StringRef Expected = "seq()";
  EXPECT_EQ(S->toString(), Expected);
}

TEST(StencilToStringTest, SequenceSingle) {
  auto S = cat("foo");
  StringRef Expected = "\"foo\"";
  EXPECT_EQ(S->toString(), Expected);
}

TEST(StencilToStringTest, SequenceFromVector) {
  auto S = catVector({cat("foo"), access("x", "m()"), cat("bar"),
                      ifBound("x", cat("t"), access("e", "f"))});
  StringRef Expected = R"repr(seq("foo", access("x", "m()"), "bar", )repr"
                       R"repr(ifBound("x", "t", access("e", "f"))))repr";
  EXPECT_EQ(S->toString(), Expected);
}
} // namespace
