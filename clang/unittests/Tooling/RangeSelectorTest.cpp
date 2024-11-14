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
#include "clang/Tooling/Transformer/Parsing.h"
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

  TraversalKindScope RAII(Context, TK_AsIs);
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
  StringRef CallID = "call";
  ast_matchers::internal::Matcher<Stmt> M = callExpr().bind(CallID);
  RangeSelector R = before(node(CallID.str()));

  TestMatch Match = matchCode(Code, M);
  const auto *E = Match.Result.Nodes.getNodeAs<Expr>(CallID);
  assert(E != nullptr);
  auto ExprBegin = E->getSourceRange().getBegin();
  EXPECT_THAT_EXPECTED(
      R(Match.Result),
      HasValue(EqualsCharSourceRange(
          CharSourceRange::getCharRange(ExprBegin, ExprBegin))));
}

TEST(RangeSelectorTest, BeforeOpParsed) {
  StringRef Code = R"cc(
    int f(int x, int y, int z) { return 3; }
    int g() { return f(/* comment */ 3, 7 /* comment */, 9); }
  )cc";
  StringRef CallID = "call";
  ast_matchers::internal::Matcher<Stmt> M = callExpr().bind(CallID);
  auto R = parseRangeSelector(R"rs(before(node("call")))rs");
  ASSERT_THAT_EXPECTED(R, llvm::Succeeded());

  TestMatch Match = matchCode(Code, M);
  const auto *E = Match.Result.Nodes.getNodeAs<Expr>(CallID);
  assert(E != nullptr);
  auto ExprBegin = E->getSourceRange().getBegin();
  EXPECT_THAT_EXPECTED(
      (*R)(Match.Result),
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

// Gets the spelling location `Length` characters after the start of AST node
// `Id`.
static SourceLocation getSpellingLocAfter(const MatchResult &Result,
                                          StringRef Id, int Length) {
  const auto *E = Result.Nodes.getNodeAs<Expr>(Id);
  assert(E != nullptr);
  return Result.SourceManager->getSpellingLoc(E->getBeginLoc())
      .getLocWithOffset(Length);
}

// Test with a range that is the entire macro arg, but does not end the
// expansion itself.
TEST(RangeSelectorTest, AfterOpInMacroArg) {
  StringRef Code = R"cc(
#define ISNULL(x) x == nullptr
    bool g() { int* y; return ISNULL(y); }
  )cc";

  TestMatch Match =
      matchCode(Code, declRefExpr(to(namedDecl(hasName("y")))).bind("yvar"));
  int YVarLen = 1;
  SourceLocation After = getSpellingLocAfter(Match.Result, "yvar", YVarLen);
  CharSourceRange Expected = CharSourceRange::getCharRange(After, After);
  EXPECT_THAT_EXPECTED(after(node("yvar"))(Match.Result),
                       HasValue(EqualsCharSourceRange(Expected)));
}

// Test with a range that is the entire macro arg and ends the expansion itself.
TEST(RangeSelectorTest, AfterOpInMacroArgEndsExpansion) {
  StringRef Code = R"cc(
#define ISNULL(x) nullptr == x
    bool g() { int* y; return ISNULL(y); }
  )cc";

  TestMatch Match =
      matchCode(Code, declRefExpr(to(namedDecl(hasName("y")))).bind("yvar"));
  int YVarLen = 1;
  SourceLocation After = getSpellingLocAfter(Match.Result, "yvar", YVarLen);
  CharSourceRange Expected = CharSourceRange::getCharRange(After, After);
  EXPECT_THAT_EXPECTED(after(node("yvar"))(Match.Result),
                       HasValue(EqualsCharSourceRange(Expected)));
}

TEST(RangeSelectorTest, AfterOpInPartOfMacroArg) {
  StringRef Code = R"cc(
#define ISNULL(x) x == nullptr
    int* f(int*);
    bool g() { int* y; return ISNULL(f(y)); }
  )cc";

  TestMatch Match =
      matchCode(Code, declRefExpr(to(namedDecl(hasName("y")))).bind("yvar"));
  int YVarLen = 1;
  SourceLocation After = getSpellingLocAfter(Match.Result, "yvar", YVarLen);
  CharSourceRange Expected = CharSourceRange::getCharRange(After, After);
  EXPECT_THAT_EXPECTED(after(node("yvar"))(Match.Result),
                       HasValue(EqualsCharSourceRange(Expected)));
}

TEST(RangeSelectorTest, BetweenOp) {
  StringRef Code = R"cc(
    int f(int x, int y, int z) { return 3; }
    int g() { return f(3, /* comment */ 7 /* comment */, 9); }
  )cc";
  auto Matcher = callExpr(hasArgument(0, expr().bind("a0")),
                          hasArgument(1, expr().bind("a1")));
  RangeSelector R = between(node("a0"), node("a1"));
  TestMatch Match = matchCode(Code, Matcher);
  EXPECT_THAT_EXPECTED(select(R, Match), HasValue(", /* comment */ "));
}

TEST(RangeSelectorTest, BetweenOpParsed) {
  StringRef Code = R"cc(
    int f(int x, int y, int z) { return 3; }
    int g() { return f(3, /* comment */ 7 /* comment */, 9); }
  )cc";
  auto Matcher = callExpr(hasArgument(0, expr().bind("a0")),
                          hasArgument(1, expr().bind("a1")));
  auto R = parseRangeSelector(R"rs(between(node("a0"), node("a1")))rs");
  ASSERT_THAT_EXPECTED(R, llvm::Succeeded());
  TestMatch Match = matchCode(Code, Matcher);
  EXPECT_THAT_EXPECTED(select(*R, Match), HasValue(", /* comment */ "));
}

// Node-id specific version.
TEST(RangeSelectorTest, EncloseOpNodes) {
  StringRef Code = R"cc(
    int f(int x, int y, int z) { return 3; }
    int g() { return f(/* comment */ 3, 7 /* comment */, 9); }
  )cc";
  auto Matcher = callExpr(hasArgument(0, expr().bind("a0")),
                          hasArgument(1, expr().bind("a1")));
  RangeSelector R = encloseNodes("a0", "a1");
  TestMatch Match = matchCode(Code, Matcher);
  EXPECT_THAT_EXPECTED(select(R, Match), HasValue("3, 7"));
}

TEST(RangeSelectorTest, EncloseOpGeneral) {
  StringRef Code = R"cc(
    int f(int x, int y, int z) { return 3; }
    int g() { return f(/* comment */ 3, 7 /* comment */, 9); }
  )cc";
  auto Matcher = callExpr(hasArgument(0, expr().bind("a0")),
                          hasArgument(1, expr().bind("a1")));
  RangeSelector R = enclose(node("a0"), node("a1"));
  TestMatch Match = matchCode(Code, Matcher);
  EXPECT_THAT_EXPECTED(select(R, Match), HasValue("3, 7"));
}

TEST(RangeSelectorTest, EncloseOpNodesParsed) {
  StringRef Code = R"cc(
    int f(int x, int y, int z) { return 3; }
    int g() { return f(/* comment */ 3, 7 /* comment */, 9); }
  )cc";
  auto Matcher = callExpr(hasArgument(0, expr().bind("a0")),
                          hasArgument(1, expr().bind("a1")));
  auto R = parseRangeSelector(R"rs(encloseNodes("a0", "a1"))rs");
  ASSERT_THAT_EXPECTED(R, llvm::Succeeded());
  TestMatch Match = matchCode(Code, Matcher);
  EXPECT_THAT_EXPECTED(select(*R, Match), HasValue("3, 7"));
}

TEST(RangeSelectorTest, EncloseOpGeneralParsed) {
  StringRef Code = R"cc(
    int f(int x, int y, int z) { return 3; }
    int g() { return f(/* comment */ 3, 7 /* comment */, 9); }
  )cc";
  auto Matcher = callExpr(hasArgument(0, expr().bind("a0")),
                          hasArgument(1, expr().bind("a1")));
  auto R = parseRangeSelector(R"rs(encloseNodes("a0", "a1"))rs");
  ASSERT_THAT_EXPECTED(R, llvm::Succeeded());
  TestMatch Match = matchCode(Code, Matcher);
  EXPECT_THAT_EXPECTED(select(*R, Match), HasValue("3, 7"));
}

TEST(RangeSelectorTest, NodeOpStatement) {
  StringRef Code = "int f() { return 3; }";
  TestMatch Match = matchCode(Code, returnStmt().bind("id"));
  EXPECT_THAT_EXPECTED(select(node("id"), Match), HasValue("return 3;"));
}

TEST(RangeSelectorTest, NodeOpExpression) {
  StringRef Code = "int f() { return 3; }";
  TestMatch Match = matchCode(Code, expr().bind("id"));
  EXPECT_THAT_EXPECTED(select(node("id"), Match), HasValue("3"));
}

TEST(RangeSelectorTest, StatementOp) {
  StringRef Code = "int f() { return 3; }";
  TestMatch Match = matchCode(Code, expr().bind("id"));
  RangeSelector R = statement("id");
  EXPECT_THAT_EXPECTED(select(R, Match), HasValue("3;"));
}

TEST(RangeSelectorTest, StatementOpParsed) {
  StringRef Code = "int f() { return 3; }";
  TestMatch Match = matchCode(Code, expr().bind("id"));
  auto R = parseRangeSelector(R"rs(statement("id"))rs");
  ASSERT_THAT_EXPECTED(R, llvm::Succeeded());
  EXPECT_THAT_EXPECTED(select(*R, Match), HasValue("3;"));
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

TEST(RangeSelectorTest, NameOpTypeLoc) {
  StringRef Code = R"cc(
    namespace ns {
    struct Foo {
      Foo();
      Foo(int);
      Foo(int, int);
    };
    }  // namespace ns

    ns::Foo a;
    auto b = ns::Foo(3);
    auto c = ns::Foo(1, 2);
  )cc";
  const char *CtorTy = "ctor_ty";
  // Matches declaration of `a`
  TestMatch MatchA = matchCode(
      Code, varDecl(hasName("a"), hasTypeLoc(typeLoc().bind(CtorTy))));
  EXPECT_THAT_EXPECTED(select(name(CtorTy), MatchA), HasValue("Foo"));
  // Matches call of Foo(int)
  TestMatch MatchB = matchCode(
      Code, cxxFunctionalCastExpr(hasTypeLoc(typeLoc().bind(CtorTy))));
  EXPECT_THAT_EXPECTED(select(name(CtorTy), MatchB), HasValue("Foo"));
  // Matches call of Foo(int, int)
  TestMatch MatchC = matchCode(
      Code, cxxTemporaryObjectExpr(hasTypeLoc(typeLoc().bind(CtorTy))));
  EXPECT_THAT_EXPECTED(select(name(CtorTy), MatchC), HasValue("Foo"));
}

TEST(RangeSelectorTest, NameOpTemplateSpecializationTypeLoc) {
  StringRef Code = R"cc(
    namespace ns {
    template <typename T>
    struct Foo {};
    }  // namespace ns

    ns::Foo<int> a;
  )cc";
  const char *Loc = "tyloc";
  // Matches declaration of `a`.
  TestMatch MatchA =
      matchCode(Code, varDecl(hasName("a"), hasTypeLoc(typeLoc().bind(Loc))));
  EXPECT_THAT_EXPECTED(select(name(Loc), MatchA), HasValue("Foo"));
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

TEST(RangeSelectorTest, ConstructExprArgs) {
  const StringRef Code = R"cc(
    struct C {
      C(int, int);
    };
    C f() {
      return C(1, 2);
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, cxxTemporaryObjectExpr().bind(ID));
  EXPECT_THAT_EXPECTED(select(constructExprArgs(ID), Match), HasValue("1, 2"));
}

TEST(RangeSelectorTest, ConstructExprBracedArgs) {
  const StringRef Code = R"cc(
    struct C {
      C(int, int);
    };
    C f() {
      return {1, 2};
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, cxxConstructExpr().bind(ID));
  EXPECT_THAT_EXPECTED(select(constructExprArgs(ID), Match), HasValue("1, 2"));
}

TEST(RangeSelectorTest, ConstructExprNoArgs) {
  const StringRef Code = R"cc(
    struct C {
      C();
    };
    C f() {
      return C();
    }
  )cc";
  const char *ID = "id";
  TestMatch Match = matchCode(Code, cxxTemporaryObjectExpr().bind(ID));
  EXPECT_THAT_EXPECTED(select(constructExprArgs(ID), Match), HasValue(""));
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
