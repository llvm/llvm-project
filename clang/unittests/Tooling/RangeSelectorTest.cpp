//===- unittest/Tooling/RangeSelectorTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Transformer/RangeSelector.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/Transformer/SourceCode.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace transformer;
using namespace ast_matchers;

namespace {
using ::llvm::Expected;
using ::llvm::Failed;
using ::llvm::HasValue;
using ::llvm::StringError;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::Property;

using MatchResult = MatchFinder::MatchResult;

struct TestMatch {
  // The AST unit from which `result` is built. We bundle it because it backs
  // the result. Users are not expected to access it.
  std::unique_ptr<clang::ASTUnit> ASTUnit;
  // The result to use in the test. References `ast_unit`.
  MatchResult Result;
};

template <typename M> TestMatch matchCode(StringRef Code, M Matcher) {
  auto ASTUnit = tooling::buildASTFromCode(Code);
  assert(ASTUnit != nullptr && "AST construction failed");

  ASTContext &Context = ASTUnit->getASTContext();
  assert(!Context.getDiagnostics().hasErrorOccurred() && "Compilation error");

  TraversalKindScope RAII(Context, ast_type_traits::TK_AsIs);
  auto Matches = ast_matchers::match(Matcher, Context);
  // We expect a single, exact match.
  assert(Matches.size() != 0 && "no matches found");
  assert(Matches.size() == 1 && "too many matches");

  return TestMatch{std::move(ASTUnit), MatchResult(Matches[0], &Context)};
}

// Applies \p Selector to \p Match and, on success, returns the selected source.
Expected<StringRef> select(RangeSelector Selector, const TestMatch &Match) {
  Expected<CharSourceRange> Range = Selector(Match.Result);
  if (!Range)
    return Range.takeError();
  return tooling::getText(*Range, *Match.Result.Context);
}

// Applies \p Selector to a trivial match with only a single bound node with id
// "bound_node_id".  For use in testing unbound-node errors.
Expected<CharSourceRange> selectFromTrivial(const RangeSelector &Selector) {
  // We need to bind the result to something, or the match will fail. Use a
  // binding that is not used in the unbound node tests.
  TestMatch Match =
      matchCode("static int x = 0;", varDecl().bind("bound_node_id"));
  return Selector(Match.Result);
}

// Matches the message expected for unbound-node failures.
testing::Matcher<StringError> withUnboundNodeMessage() {
  return testing::Property(
      &StringError::getMessage,
      AllOf(HasSubstr("unbound_id"), HasSubstr("not bound")));
}

// Applies \p Selector to code containing assorted node types, where the match
// binds each one: a statement ("stmt"), a (non-member) ctor-initializer
// ("init"), an expression ("expr") and a (nameless) declaration ("decl").  Used
// to test failures caused by applying selectors to nodes of the wrong type.
Expected<CharSourceRange> selectFromAssorted(RangeSelector Selector) {
  StringRef Code = R"cc(
      struct A {};
      class F : public A {
       public:
        F(int) {}
      };
      void g() { F f(1); }
    )cc";

  auto Matcher =
      compoundStmt(
          hasDescendant(
              cxxConstructExpr(
                  hasDeclaration(
                      decl(hasDescendant(cxxCtorInitializer(isBaseInitializer())
                                             .bind("init")))
                          .bind("decl")))
                  .bind("expr")))
          .bind("stmt");

  return Selector(matchCode(Code, Matcher).Result);
}

// Matches the message expected for type-error failures.
testing::Matcher<StringError> withTypeErrorMessage(const std::string &NodeID) {
  return testing::Property(
      &StringError::getMessage,
      AllOf(HasSubstr(NodeID), HasSubstr("mismatched type")));
}

TEST(RangeSelectorTest, UnboundNode) {
  EXPECT_THAT_EXPECTED(selectFromTrivial(node("unbound_id")),
                       Failed<StringError>(withUnboundNodeMessage()));
}

MATCHER_P(EqualsCharSourceRange, Range, "") {
  return Range.getAsRange() == arg.getAsRange() &&
         Range.isTokenRange() == arg.isTokenRange();
}

// FIXME: here and elsewhere: use llvm::Annotations library to explicitly mark
// points and ranges of interest, enabling more readable tests.
TEST(RangeSelectorTest, BeforeOp) {
  StringRef Code = R"cc(
    int f(int x, int y, int z) { return 3; }
    int g() { return f(/* comment */ 3, 7 /* comment */, 9); }
  )cc";
  const char *Call = "call";
  TestMatch Match = matchCode(Code, callExpr().bind(Call));
  const auto* E = Match.Result.Nodes.getNodeAs<Expr>(Call);
  assert(E != nullptr);
  auto ExprBegin = E->getSourceRange().getBegin();
  EXPECT_THAT_EXPECTED(
      before(node(Call))(Match.Result),
      HasValue(EqualsCharSourceRange(
          CharSourceRange::getCharRange(ExprBegin, ExprBegin))));
}

TEST(RangeSelectorTest, AfterOp) {
  StringRef Code = R"cc(
    int f(int x, int y, int z) { return 3; }
    int g() { return f(/* comment */ 3, 7 /* comment */, 9); }
  )cc";
  StringRef Call = "call";
  TestMatch Match = matchCode(Code, callExpr().bind(Call));
  const auto* E = Match.Result.Nodes.getNodeAs<Expr>(Call);
  assert(E != nullptr);
  const SourceRange Range = E->getSourceRange();
  // The end token, a right paren, is one character wide, so advance by one,
  // bringing us to the semicolon.
  const SourceLocation SemiLoc = Range.getEnd().getLocWithOffset(1);
  const auto ExpectedAfter = CharSourceRange::getCharRange(SemiLoc, SemiLoc);

  // Test with a char range.
  auto CharRange = CharSourceRange::getCharRange(Range.getBegin(), SemiLoc);
  EXPECT_THAT_EXPECTED(after(charRange(CharRange))(Match.Result),
                       HasValue(EqualsCharSourceRange(ExpectedAfter)));

  // Test with a token range.
  auto TokenRange = CharSourceRange::getTokenRange(Range);
  EXPECT_THAT_EXPECTED(after(charRange(TokenRange))(Match.Result),
                       HasValue(EqualsCharSourceRange(ExpectedAfter)));
}

TEST(RangeSelectorTest, RangeOp) {
  StringRef Code = R"cc(
    int f(int x, int y, int z) { return 3; }
    int g() { return f(/* comment */ 3, 7 /* comment */, 9); }
  )cc";
  const char *Arg0 = "a0";
  const char *Arg1 = "a1";
  StringRef Call = "call";
  auto Matcher = callExpr(hasArgument(0, expr().bind(Arg0)),
                          hasArgument(1, expr().bind(Arg1)))
                     .bind(Call);
  TestMatch Match = matchCode(Code, Matcher);

  // Node-id specific version:
  EXPECT_THAT_EXPECTED(select(range(Arg0, Arg1), Match), HasValue("3, 7"));
  // General version:
  EXPECT_THAT_EXPECTED(select(range(node(Arg0), node(Arg1)), Match),
                       HasValue("3, 7"));
}

TEST(RangeSelectorTest, NodeOpStatement) {
  StringRef Code = "int f() { return 3; }";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, returnStmt().bind(ID));
  EXPECT_THAT_EXPECTED(select(node(ID), Match), HasValue("return 3;"));
}

TEST(RangeSelectorTest, NodeOpExpression) {
  StringRef Code = "int f() { return 3; }";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, expr().bind(ID));
  EXPECT_THAT_EXPECTED(select(node(ID), Match), HasValue("3"));
}

TEST(RangeSelectorTest, StatementOp) {
  StringRef Code = "int f() { return 3; }";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, expr().bind(ID));
  EXPECT_THAT_EXPECTED(select(statement(ID), Match), HasValue("3;"));
}

TEST(RangeSelectorTest, MemberOp) {
  StringRef Code = R"cc(
    struct S {
      int member;
    };
    int g() {
      S s;
      return s.member;
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, memberExpr().bind(ID));
  EXPECT_THAT_EXPECTED(select(member(ID), Match), HasValue("member"));
}

// Tests that member does not select any qualifiers on the member name.
TEST(RangeSelectorTest, MemberOpQualified) {
  StringRef Code = R"cc(
    struct S {
      int member;
    };
    struct T : public S {
      int field;
    };
    int g() {
      T t;
      return t.S::member;
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, memberExpr().bind(ID));
  EXPECT_THAT_EXPECTED(select(member(ID), Match), HasValue("member"));
}

TEST(RangeSelectorTest, MemberOpTemplate) {
  StringRef Code = R"cc(
    struct S {
      template <typename T> T foo(T t);
    };
    int f(int x) {
      S s;
      return s.template foo<int>(3);
    }
  )cc";

  const char *ID = "id";
  TestMatch Match = matchCode(Code, memberExpr().bind(ID));
  EXPECT_THAT_EXPECTED(select(member(ID), Match), HasValue("foo"));
}

TEST(RangeSelectorTest, MemberOpOperator) {
  StringRef Code = R"cc(
    struct S {
      int operator*();
    };
    int f(int x) {
      S s;
      return s.operator *();
    }
  )cc";

  const char *ID = "id";
  TestMatch Match = matchCode(Code, memberExpr().bind(ID));
  EXPECT_THAT_EXPECTED(select(member(ID), Match), HasValue("operator *"));
}

TEST(RangeSelectorTest, NameOpNamedDecl) {
  StringRef Code = R"cc(
    int myfun() {
      return 3;
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, functionDecl().bind(ID));
  EXPECT_THAT_EXPECTED(select(name(ID), Match), HasValue("myfun"));
}

TEST(RangeSelectorTest, NameOpDeclRef) {
  StringRef Code = R"cc(
    int foo(int x) {
      return x;
    }
    int g(int x) { return foo(x) * x; }
  )cc";
  const char *Ref = "ref";
  TestMatch Match = matchCode(Code, declRefExpr(to(functionDecl())).bind(Ref));
  EXPECT_THAT_EXPECTED(select(name(Ref), Match), HasValue("foo"));
}

TEST(RangeSelectorTest, NameOpCtorInitializer) {
  StringRef Code = R"cc(
    class C {
     public:
      C() : field(3) {}
      int field;
    };
  )cc";
  const char *Init = "init";
  TestMatch Match = matchCode(Code, cxxCtorInitializer().bind(Init));
  EXPECT_THAT_EXPECTED(select(name(Init), Match), HasValue("field"));
}

TEST(RangeSelectorTest, NameOpErrors) {
  EXPECT_THAT_EXPECTED(selectFromTrivial(name("unbound_id")),
                       Failed<StringError>(withUnboundNodeMessage()));
  EXPECT_THAT_EXPECTED(selectFromAssorted(name("stmt")),
                       Failed<StringError>(withTypeErrorMessage("stmt")));
}

TEST(RangeSelectorTest, NameOpDeclRefError) {
  StringRef Code = R"cc(
    struct S {
      int operator*();
    };
    int f(int x) {
      S s;
      return *s + x;
    }
  )cc";
  const char *Ref = "ref";
  TestMatch Match = matchCode(Code, declRefExpr(to(functionDecl())).bind(Ref));
  EXPECT_THAT_EXPECTED(
      name(Ref)(Match.Result),
      Failed<StringError>(testing::Property(
          &StringError::getMessage,
          AllOf(HasSubstr(Ref), HasSubstr("requires property 'identifier'")))));
}

TEST(RangeSelectorTest, CallArgsOp) {
  const StringRef Code = R"cc(
    struct C {
      int bar(int, int);
    };
    int f() {
      C x;
      return x.bar(3, 4);
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, callExpr().bind(ID));
  EXPECT_THAT_EXPECTED(select(callArgs(ID), Match), HasValue("3, 4"));
}

TEST(RangeSelectorTest, CallArgsOpNoArgs) {
  const StringRef Code = R"cc(
    struct C {
      int bar();
    };
    int f() {
      C x;
      return x.bar();
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, callExpr().bind(ID));
  EXPECT_THAT_EXPECTED(select(callArgs(ID), Match), HasValue(""));
}

TEST(RangeSelectorTest, CallArgsOpNoArgsWithComments) {
  const StringRef Code = R"cc(
    struct C {
      int bar();
    };
    int f() {
      C x;
      return x.bar(/*empty*/);
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, callExpr().bind(ID));
  EXPECT_THAT_EXPECTED(select(callArgs(ID), Match), HasValue("/*empty*/"));
}

// Tests that arguments are extracted correctly when a temporary (with parens)
// is used.
TEST(RangeSelectorTest, CallArgsOpWithParens) {
  const StringRef Code = R"cc(
    struct C {
      int bar(int, int) { return 3; }
    };
    int f() {
      C x;
      return C().bar(3, 4);
    }
  )cc";
  const char *ID = "id";
  TestMatch Match =
      matchCode(Code, callExpr(callee(functionDecl(hasName("bar")))).bind(ID));
  EXPECT_THAT_EXPECTED(select(callArgs(ID), Match), HasValue("3, 4"));
}

TEST(RangeSelectorTest, CallArgsOpLeadingComments) {
  const StringRef Code = R"cc(
    struct C {
      int bar(int, int) { return 3; }
    };
    int f() {
      C x;
      return x.bar(/*leading*/ 3, 4);
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, callExpr().bind(ID));
  EXPECT_THAT_EXPECTED(select(callArgs(ID), Match),
                       HasValue("/*leading*/ 3, 4"));
}

TEST(RangeSelectorTest, CallArgsOpTrailingComments) {
  const StringRef Code = R"cc(
    struct C {
      int bar(int, int) { return 3; }
    };
    int f() {
      C x;
      return x.bar(3 /*trailing*/, 4);
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, callExpr().bind(ID));
  EXPECT_THAT_EXPECTED(select(callArgs(ID), Match),
                       HasValue("3 /*trailing*/, 4"));
}

TEST(RangeSelectorTest, CallArgsOpEolComments) {
  const StringRef Code = R"cc(
    struct C {
      int bar(int, int) { return 3; }
    };
    int f() {
      C x;
      return x.bar(  // Header
          1,           // foo
          2            // bar
      );
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, callExpr().bind(ID));
  std::string ExpectedString = R"(  // Header
          1,           // foo
          2            // bar
      )";
  EXPECT_THAT_EXPECTED(select(callArgs(ID), Match), HasValue(ExpectedString));
}

TEST(RangeSelectorTest, CallArgsErrors) {
  EXPECT_THAT_EXPECTED(selectFromTrivial(callArgs("unbound_id")),
                       Failed<StringError>(withUnboundNodeMessage()));
  EXPECT_THAT_EXPECTED(selectFromAssorted(callArgs("stmt")),
                       Failed<StringError>(withTypeErrorMessage("stmt")));
}

TEST(RangeSelectorTest, StatementsOp) {
  StringRef Code = R"cc(
    void g();
    void f() { /* comment */ g(); /* comment */ g(); /* comment */ }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, compoundStmt().bind(ID));
  EXPECT_THAT_EXPECTED(
      select(statements(ID), Match),
      HasValue(" /* comment */ g(); /* comment */ g(); /* comment */ "));
}

TEST(RangeSelectorTest, StatementsOpEmptyList) {
  StringRef Code = "void f() {}";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, compoundStmt().bind(ID));
  EXPECT_THAT_EXPECTED(select(statements(ID), Match), HasValue(""));
}

TEST(RangeSelectorTest, StatementsOpErrors) {
  EXPECT_THAT_EXPECTED(selectFromTrivial(statements("unbound_id")),
                       Failed<StringError>(withUnboundNodeMessage()));
  EXPECT_THAT_EXPECTED(selectFromAssorted(statements("decl")),
                       Failed<StringError>(withTypeErrorMessage("decl")));
}

TEST(RangeSelectorTest, ElementsOp) {
  StringRef Code = R"cc(
    void f() {
      int v[] = {/* comment */ 3, /* comment*/ 4 /* comment */};
      (void)v;
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, initListExpr().bind(ID));
  EXPECT_THAT_EXPECTED(
      select(initListElements(ID), Match),
      HasValue("/* comment */ 3, /* comment*/ 4 /* comment */"));
}

TEST(RangeSelectorTest, ElementsOpEmptyList) {
  StringRef Code = R"cc(
    void f() {
      int v[] = {};
      (void)v;
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, initListExpr().bind(ID));
  EXPECT_THAT_EXPECTED(select(initListElements(ID), Match), HasValue(""));
}

TEST(RangeSelectorTest, ElementsOpErrors) {
  EXPECT_THAT_EXPECTED(selectFromTrivial(initListElements("unbound_id")),
                       Failed<StringError>(withUnboundNodeMessage()));
  EXPECT_THAT_EXPECTED(selectFromAssorted(initListElements("stmt")),
                       Failed<StringError>(withTypeErrorMessage("stmt")));
}

TEST(RangeSelectorTest, ElseBranchOpSingleStatement) {
  StringRef Code = R"cc(
    int f() {
      int x = 0;
      if (true) x = 3;
      else x = 4;
      return x + 5;
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, ifStmt().bind(ID));
  EXPECT_THAT_EXPECTED(select(elseBranch(ID), Match), HasValue("else x = 4;"));
}

TEST(RangeSelectorTest, ElseBranchOpCompoundStatement) {
  StringRef Code = R"cc(
    int f() {
      int x = 0;
      if (true) x = 3;
      else { x = 4; }
      return x + 5;
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, ifStmt().bind(ID));
  EXPECT_THAT_EXPECTED(select(elseBranch(ID), Match),
                       HasValue("else { x = 4; }"));
}

// Tests case where the matched node is the complete expanded text.
TEST(RangeSelectorTest, ExpansionOp) {
  StringRef Code = R"cc(
#define BADDECL(E) int bad(int x) { return E; }
    BADDECL(x * x)
  )cc";

  const char *Fun = "Fun";
  TestMatch Match = matchCode(Code, functionDecl(hasName("bad")).bind(Fun));
  EXPECT_THAT_EXPECTED(select(expansion(node(Fun)), Match),
                       HasValue("BADDECL(x * x)"));
}

// Tests case where the matched node is (only) part of the expanded text.
TEST(RangeSelectorTest, ExpansionOpPartial) {
  StringRef Code = R"cc(
#define BADDECL(E) int bad(int x) { return E; }
    BADDECL(x * x)
  )cc";

  const char *Ret = "Ret";
  TestMatch Match = matchCode(Code, returnStmt().bind(Ret));
  EXPECT_THAT_EXPECTED(select(expansion(node(Ret)), Match),
                       HasValue("BADDECL(x * x)"));
}

TEST(RangeSelectorTest, IfBoundOpBound) {
  StringRef Code = R"cc(
    int f() {
      return 3 + 5;
    }
  )cc";
  const char *ID = "id", *Op = "op";
  TestMatch Match =
      matchCode(Code, binaryOperator(hasLHS(expr().bind(ID))).bind(Op));
  EXPECT_THAT_EXPECTED(select(ifBound(ID, node(ID), node(Op)), Match),
                       HasValue("3"));
}

TEST(RangeSelectorTest, IfBoundOpUnbound) {
  StringRef Code = R"cc(
    int f() {
      return 3 + 5;
    }
  )cc";
  const char *ID = "id", *Op = "op";
  TestMatch Match = matchCode(Code, binaryOperator().bind(Op));
  EXPECT_THAT_EXPECTED(select(ifBound(ID, node(ID), node(Op)), Match),
                       HasValue("3 + 5"));
}

} // namespace
