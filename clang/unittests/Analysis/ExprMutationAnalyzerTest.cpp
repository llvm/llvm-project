//===---------- ExprMutationAnalyzerTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cctype>

namespace clang {

using namespace clang::ast_matchers;
using ::testing::ElementsAre;
using ::testing::ResultOf;
using ::testing::Values;

namespace {

using ExprMatcher = internal::Matcher<Expr>;
using StmtMatcher = internal::Matcher<Stmt>;

std::unique_ptr<ASTUnit>
buildASTFromCodeWithArgs(const Twine &Code,
                         const std::vector<std::string> &Args) {
  SmallString<1024> CodeStorage;
  auto AST =
      tooling::buildASTFromCodeWithArgs(Code.toStringRef(CodeStorage), Args);
  EXPECT_FALSE(AST->getDiagnostics().hasErrorOccurred());
  return AST;
}

std::unique_ptr<ASTUnit> buildASTFromCode(const Twine &Code) {
  return buildASTFromCodeWithArgs(Code, {});
}

ExprMatcher declRefTo(StringRef Name) {
  return declRefExpr(to(namedDecl(hasName(Name))));
}

StmtMatcher withEnclosingCompound(ExprMatcher Matcher) {
  return expr(Matcher, hasAncestor(compoundStmt().bind("stmt"))).bind("expr");
}

bool isMutated(const SmallVectorImpl<BoundNodes> &Results, ASTUnit *AST) {
  const auto *const S = selectFirst<Stmt>("stmt", Results);
  const auto *const E = selectFirst<Expr>("expr", Results);
  TraversalKindScope RAII(AST->getASTContext(), TK_AsIs);
  return ExprMutationAnalyzer(*S, AST->getASTContext()).isMutated(E);
}

SmallVector<std::string, 1>
mutatedBy(const SmallVectorImpl<BoundNodes> &Results, ASTUnit *AST) {
  const auto *const S = selectFirst<Stmt>("stmt", Results);
  SmallVector<std::string, 1> Chain;
  ExprMutationAnalyzer Analyzer(*S, AST->getASTContext());

  for (const auto *E = selectFirst<Expr>("expr", Results); E != nullptr;) {
    const Stmt *By = Analyzer.findMutation(E);
    if (!By)
      break;

    std::string Buffer;
    llvm::raw_string_ostream Stream(Buffer);
    By->printPretty(Stream, nullptr, AST->getASTContext().getPrintingPolicy());
    Chain.emplace_back(StringRef(Stream.str()).trim().str());
    E = dyn_cast<DeclRefExpr>(By);
  }
  return Chain;
}

std::string removeSpace(std::string s) {
  s.erase(std::remove_if(s.begin(), s.end(),
                         [](char c) { return llvm::isSpace(c); }),
          s.end());
  return s;
}

const std::string StdRemoveReference =
    "namespace std {"
    "template<class T> struct remove_reference { typedef T type; };"
    "template<class T> struct remove_reference<T&> { typedef T type; };"
    "template<class T> struct remove_reference<T&&> { typedef T type; }; }";

const std::string StdMove =
    "namespace std {"
    "template<class T> typename remove_reference<T>::type&& "
    "move(T&& t) noexcept {"
    "return static_cast<typename remove_reference<T>::type&&>(t); } }";

const std::string StdForward =
    "namespace std {"
    "template<class T> T&& "
    "forward(typename remove_reference<T>::type& t) noexcept { return t; }"
    "template<class T> T&& "
    "forward(typename remove_reference<T>::type&& t) noexcept { return t; } }";

} // namespace

TEST(ExprMutationAnalyzerTest, Trivial) {
  const auto AST = buildASTFromCode("void f() { int x; x; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

class AssignmentTest : public ::testing::TestWithParam<std::string> {};

// This test is for the most basic and direct modification of a variable,
// assignment to it (e.g. `x = 10;`).
// It additionally tests that references to a variable are not only captured
// directly but expressions that result in the variable are handled, too.
// This includes the comma operator, parens and the ternary operator.
TEST_P(AssignmentTest, AssignmentModifies) {
  // Test the detection of the raw expression modifications.
  {
    const std::string ModExpr = "x " + GetParam() + " 10";
    const auto AST = buildASTFromCode("void f() { int x; " + ModExpr + "; }");
    const auto Results =
        match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
    EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre(ModExpr));
  }

  // Test the detection if the expression is surrounded by parens.
  {
    const std::string ModExpr = "(x) " + GetParam() + " 10";
    const auto AST = buildASTFromCode("void f() { int x; " + ModExpr + "; }");
    const auto Results =
        match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
    EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre(ModExpr));
  }

  // Test the detection if the comma operator yields the expression as result.
  {
    const std::string ModExpr = "x " + GetParam() + " 10";
    const auto AST = buildASTFromCodeWithArgs(
        "void f() { int x, y; y, " + ModExpr + "; }", {"-Wno-unused-value"});
    const auto Results =
        match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
    EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre(ModExpr));
  }

  // Ensure no detection if the comma operator does not yield the expression as
  // result.
  {
    const std::string ModExpr = "y, x, y " + GetParam() + " 10";
    const auto AST = buildASTFromCodeWithArgs(
        "void f() { int x, y; " + ModExpr + "; }", {"-Wno-unused-value"});
    const auto Results =
        match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
    EXPECT_FALSE(isMutated(Results, AST.get()));
  }

  // Test the detection if the a ternary operator can result in the expression.
  {
    const std::string ModExpr = "(y != 0 ? y : x) " + GetParam() + " 10";
    const auto AST =
        buildASTFromCode("void f() { int y = 0, x; " + ModExpr + "; }");
    const auto Results =
        match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
    EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre(ModExpr));
  }

  // Test the detection if the a ternary operator can result in the expression
  // through multiple nesting of ternary operators.
  {
    const std::string ModExpr =
        "(y != 0 ? (y > 5 ? y : x) : (y)) " + GetParam() + " 10";
    const auto AST =
        buildASTFromCode("void f() { int y = 0, x; " + ModExpr + "; }");
    const auto Results =
        match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
    EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre(ModExpr));
  }

  // Test the detection if the a ternary operator can result in the expression
  // with additional parens.
  {
    const std::string ModExpr = "(y != 0 ? (y) : ((x))) " + GetParam() + " 10";
    const auto AST =
        buildASTFromCode("void f() { int y = 0, x; " + ModExpr + "; }");
    const auto Results =
        match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
    EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre(ModExpr));
  }

  // Test the detection for the binary conditional operator.
  {
    const std::string ModExpr = "(y ?: x) " + GetParam() + " 10";
    const auto AST =
        buildASTFromCode("void f() { int y = 0, x; " + ModExpr + "; }");
    const auto Results =
        match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
    EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre(ModExpr));
  }
}

INSTANTIATE_TEST_SUITE_P(AllAssignmentOperators, AssignmentTest,
                        Values("=", "+=", "-=", "*=", "/=", "%=", "&=", "|=",
                               "^=", "<<=", ">>=") );

TEST(ExprMutationAnalyzerTest, AssignmentConditionalWithInheritance) {
  const auto AST = buildASTFromCode("struct Base {void nonconst(); };"
                                    "struct Derived : Base {};"
                                    "static void f() {"
                                    "  Derived x, y;"
                                    "  Base &b = true ? x : y;"
                                    "  b.nonconst();"
                                    "}");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("b", "b.nonconst()"));
}

class IncDecTest : public ::testing::TestWithParam<std::string> {};

TEST_P(IncDecTest, IncDecModifies) {
  const std::string ModExpr = GetParam();
  const auto AST = buildASTFromCode("void f() { int x; " + ModExpr + "; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre(ModExpr));
}

INSTANTIATE_TEST_SUITE_P(AllIncDecOperators, IncDecTest,
                        Values("++x", "--x", "x++", "x--", "++(x)", "--(x)",
                               "(x)++", "(x)--") );

// Section: member functions

TEST(ExprMutationAnalyzerTest, NonConstMemberFunc) {
  const auto AST = buildASTFromCode(
      "void f() { struct Foo { void mf(); }; Foo x; x.mf(); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x.mf()"));
}

TEST(ExprMutationAnalyzerTest, AssumedNonConstMemberFunc) {
  auto AST = buildASTFromCodeWithArgs(
      "struct X { template <class T> void mf(); };"
      "template <class T> void f() { X x; x.mf<T>(); }",
      {"-fno-delayed-template-parsing"});
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x.mf<T>()"));

  AST = buildASTFromCodeWithArgs("template <class T> void f() { T x; x.mf(); }",
                                 {"-fno-delayed-template-parsing"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x.mf()"));

  AST = buildASTFromCodeWithArgs(
      "template <class T> struct X;"
      "template <class T> void f() { X<T> x; x.mf(); }",
      {"-fno-delayed-template-parsing"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x.mf()"));
}

TEST(ExprMutationAnalyzerTest, ConstMemberFunc) {
  const auto AST = buildASTFromCode(
      "void f() { struct Foo { void mf() const; }; Foo x; x.mf(); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, TypeDependentMemberCall) {
  const auto AST = buildASTFromCodeWithArgs(
      "template <class T> class vector { void push_back(T); }; "
      "template <class T> void f() { vector<T> x; x.push_back(T()); }",
      {"-fno-delayed-template-parsing"});
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x.push_back(T())"));
}

TEST(ExprMutationAnalyzerTest, MemberPointerMemberCall) {
  {
    const auto AST =
        buildASTFromCode("struct X {};"
                         "using T = int (X::*)();"
                         "void f(X &x, T m) { X &ref = x; (ref.*m)(); }");
    const auto Results =
        match(withEnclosingCompound(declRefTo("ref")), AST->getASTContext());
    EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("(ref .* m)()"));
  }
  {
    const auto AST =
        buildASTFromCode("struct X {};"
                         "using T = int (X::*)();"
                         "void f(X &x, T const m) { X &ref = x; (ref.*m)(); }");
    const auto Results =
        match(withEnclosingCompound(declRefTo("ref")), AST->getASTContext());
    EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("(ref .* m)()"));
  }
  {
    const auto AST =
        buildASTFromCode("struct X {};"
                         "using T = int (X::*)() const;"
                         "void f(X &x, T m) { X &ref = x; (ref.*m)(); }");
    const auto Results =
        match(withEnclosingCompound(declRefTo("ref")), AST->getASTContext());
    EXPECT_FALSE(isMutated(Results, AST.get()));
  }
}

// Section: overloaded operators

TEST(ExprMutationAnalyzerTest, NonConstOperator) {
  const auto AST = buildASTFromCode(
      "void f() { struct Foo { Foo& operator=(int); }; Foo x; x = 10; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x = 10"));
}

TEST(ExprMutationAnalyzerTest, ConstOperator) {
  const auto AST = buildASTFromCode(
      "void f() { struct Foo { int operator()() const; }; Foo x; x(); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, UnresolvedOperator) {
  const auto AST = buildASTFromCodeWithArgs(
      "template <typename Stream> void input_operator_template() {"
      "Stream x; unsigned y = 42;"
      "x >> y; }",
      {"-fno-delayed-template-parsing"});
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
}

// Section: expression as call argument

TEST(ExprMutationAnalyzerTest, ByValueArgument) {
  auto AST = buildASTFromCode("void g(int); void f() { int x; g(x); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("void g(int*); void f() { int* x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("typedef int* IntPtr;"
                         "void g(IntPtr); void f() { int* x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(
      "struct A {}; A operator+(A, int); void f() { A x; x + 1; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("void f() { struct A { A(int); }; int x; A y(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("struct A { A(); A& operator=(A); };"
                         "void f() { A x, y; y = x; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(
      "template <int> struct A { A(); A(const A&); static void mf(A) {} };"
      "void f() { A<0> x; A<0>::mf(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, ByConstValueArgument) {
  auto AST = buildASTFromCode("void g(const int); void f() { int x; g(x); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("void g(int* const); void f() { int* x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("typedef int* const CIntPtr;"
                         "void g(CIntPtr); void f() { int* x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(
      "struct A {}; A operator+(const A, int); void f() { A x; x + 1; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(
      "void f() { struct A { A(const int); }; int x; A y(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("template <int> struct A { A(); A(const A&);"
                         "static void mf(const A&) {} };"
                         "void f() { A<0> x; A<0>::mf(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, ByNonConstRefArgument) {
  auto AST = buildASTFromCode("void g(int&); void f() { int x; g(x); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g(x)"));

  AST = buildASTFromCode("typedef int& IntRef;"
                         "void g(IntRef); void f() { int x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g(x)"));

  AST = buildASTFromCode("template <class T> using TRef = T&;"
                         "void g(TRef<int>); void f() { int x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g(x)"));

  AST = buildASTFromCode(
      "template <class T> struct identity { using type = T; };"
      "template <class T, class U = T&> void g(typename identity<U>::type);"
      "void f() { int x; g<int>(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g<int>(x)"));

  AST = buildASTFromCode("typedef int* IntPtr;"
                         "void g(IntPtr&); void f() { int* x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g(x)"));

  AST = buildASTFromCode("typedef int* IntPtr; typedef IntPtr& IntPtrRef;"
                         "void g(IntPtrRef); void f() { int* x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g(x)"));

  AST = buildASTFromCode(
      "struct A {}; A operator+(A&, int); void f() { A x; x + 1; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x + 1"));

  AST = buildASTFromCode("void f() { struct A { A(int&); }; int x; A y(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x"));

  AST = buildASTFromCode("void f() { struct A { A(); A(A&); }; A x; A y(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x"));

  AST = buildASTFromCode(
      "template <int> struct A { A(); A(const A&); static void mf(A&) {} };"
      "void f() { A<0> x; A<0>::mf(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("A<0>::mf(x)"));
}

TEST(ExprMutationAnalyzerTest, ByNonConstRefArgumentFunctionTypeDependent) {
  auto AST = buildASTFromCodeWithArgs(
      "enum MyEnum { foo, bar };"
      "void tryParser(unsigned& first, MyEnum Type) { first++, (void)Type; }"
      "template <MyEnum Type> void parse() {"
      "  auto parser = [](unsigned& first) { first++; tryParser(first, Type); "
      "};"
      "  unsigned x = 42;"
      "  parser(x);"
      "}",
      {"-fno-delayed-template-parsing"});
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("parser(x)"));
}

TEST(ExprMutationAnalyzerTest, ByConstRefArgument) {
  auto AST = buildASTFromCode("void g(const int&); void f() { int x; g(x); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("typedef const int& CIntRef;"
                         "void g(CIntRef); void f() { int x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("template <class T> using CTRef = const T&;"
                         "void g(CTRef<int>); void f() { int x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST =
      buildASTFromCode("template <class T> struct identity { using type = T; };"
                       "template <class T, class U = const T&>"
                       "void g(typename identity<U>::type);"
                       "void f() { int x; g<int>(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(
      "struct A {}; A operator+(const A&, int); void f() { A x; x + 1; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(
      "void f() { struct A { A(const int&); }; int x; A y(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(
      "void f() { struct A { A(); A(const A&); }; A x; A y(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, ByNonConstRRefArgument) {
  auto AST = buildASTFromCode(
      "void g(int&&); void f() { int x; g(static_cast<int &&>(x)); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("g(static_cast<int &&>(x))"));

  AST = buildASTFromCode("struct A {}; A operator+(A&&, int);"
                         "void f() { A x; static_cast<A &&>(x) + 1; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<A &&>(x) + 1"));

  AST = buildASTFromCode("void f() { struct A { A(int&&); }; "
                         "int x; A y(static_cast<int &&>(x)); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<int &&>(x)"));

  AST = buildASTFromCode("void f() { struct A { A(); A(A&&); }; "
                         "A x; A y(static_cast<A &&>(x)); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<A &&>(x)"));
}

TEST(ExprMutationAnalyzerTest, ByConstRRefArgument) {
  auto AST = buildASTFromCode(
      "void g(const int&&); void f() { int x; g(static_cast<int&&>(x)); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<int &&>(x)"));

  AST = buildASTFromCode("struct A {}; A operator+(const A&&, int);"
                         "void f() { A x; static_cast<A&&>(x) + 1; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<A &&>(x)"));

  AST = buildASTFromCode("void f() { struct A { A(const int&&); }; "
                         "int x; A y(static_cast<int&&>(x)); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<int &&>(x)"));

  AST = buildASTFromCode("void f() { struct A { A(); A(const A&&); }; "
                         "A x; A y(static_cast<A&&>(x)); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<A &&>(x)"));
}

// section: explicit std::move and std::forward testing

TEST(ExprMutationAnalyzerTest, Move) {
  auto AST = buildASTFromCode(StdRemoveReference + StdMove +
                              "void f() { struct A {}; A x; std::move(x); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(StdRemoveReference + StdMove +
                         "void f() { struct A {}; A x, y; std::move(x) = y; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("std::move(x) = y"));

  AST = buildASTFromCode(StdRemoveReference + StdMove +
                         "void f() { int x, y; y = std::move(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST =
      buildASTFromCode(StdRemoveReference + StdMove +
                       "struct S { S(); S(const S&); S& operator=(const S&); };"
                       "void f() { S x, y; y = std::move(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(StdRemoveReference + StdMove +
                         "struct S { S(); S(S&&); S& operator=(S&&); };"
                         "void f() { S x, y; y = std::move(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("y = std::move(x)"));

  AST = buildASTFromCode(StdRemoveReference + StdMove +
                         "struct S { S(); S(const S&); S(S&&);"
                         "S& operator=(const S&); S& operator=(S&&); };"
                         "void f() { S x, y; y = std::move(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("y = std::move(x)"));

  AST = buildASTFromCode(StdRemoveReference + StdMove +
                         "struct S { S(); S(const S&); S(S&&);"
                         "S& operator=(const S&); S& operator=(S&&); };"
                         "void f() { const S x; S y; y = std::move(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(StdRemoveReference + StdMove +
                         "struct S { S(); S& operator=(S); };"
                         "void f() { S x, y; y = std::move(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(StdRemoveReference + StdMove +
                         "struct S{}; void f() { S x, y; y = std::move(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("y = std::move(x)"));

  AST = buildASTFromCode(
      StdRemoveReference + StdMove +
      "struct S{}; void f() { const S x; S y; y = std::move(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, Forward) {
  auto AST =
      buildASTFromCode(StdRemoveReference + StdForward +
                       "void f() { struct A {}; A x; std::forward<A &>(x); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(
      StdRemoveReference + StdForward +
      "void f() { struct A {}; A x, y; std::forward<A &>(x) = y; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("std::forward<A &>(x) = y"));
}

// section: template constellations that prohibit reasoning about modifications
//          as it depends on instantiations.

TEST(ExprMutationAnalyzerTest, CallUnresolved) {
  auto AST =
      buildASTFromCodeWithArgs("template <class T> void f() { T x; g(x); }",
                               {"-fno-delayed-template-parsing"});
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g(x)"));

  AST =
      buildASTFromCodeWithArgs("template <int N> void f() { char x[N]; g(x); }",
                               {"-fno-delayed-template-parsing"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g(x)"));

  AST = buildASTFromCodeWithArgs(
      "template <class T> void f(T t) { int x; g(t, x); }",
      {"-fno-delayed-template-parsing"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g(t, x)"));

  AST = buildASTFromCodeWithArgs(
      "template <class T> void f(T t) { int x; t.mf(x); }",
      {"-fno-delayed-template-parsing"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("t.mf(x)"));

  AST = buildASTFromCodeWithArgs(
      "template <class T> struct S;"
      "template <class T> void f() { S<T> s; int x; s.mf(x); }",
      {"-fno-delayed-template-parsing"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("s.mf(x)"));

  AST = buildASTFromCodeWithArgs(
      "struct S { template <class T> void mf(); };"
      "template <class T> void f(S s) { int x; s.mf<T>(x); }",
      {"-fno-delayed-template-parsing"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("s.mf<T>(x)"));

  AST = buildASTFromCodeWithArgs("template <class F>"
                                 "void g(F f) { int x; f(x); } ",
                                 {"-fno-delayed-template-parsing"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("f(x)"));

  AST = buildASTFromCodeWithArgs(
      "template <class T> void f() { int x; (void)T(x); }",
      {"-fno-delayed-template-parsing"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("T(x)"));
}

// section: return values

TEST(ExprMutationAnalyzerTest, ReturnAsValue) {
  auto AST = buildASTFromCode("int f() { int x; return x; }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("int* f() { int* x; return x; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("typedef int* IntPtr;"
                         "IntPtr f() { int* x; return x; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, ReturnAsNonConstRef) {
  const auto AST = buildASTFromCode("int& f() { int x; return x; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("return x;"));
}

TEST(ExprMutationAnalyzerTest, ReturnAsConstRef) {
  const auto AST = buildASTFromCode("const int& f() { int x; return x; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, ReturnAsNonConstRRef) {
  const auto AST =
      buildASTFromCode("int&& f() { int x; return static_cast<int &&>(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<int &&>(x)"));
}

TEST(ExprMutationAnalyzerTest, ReturnAsConstRRef) {
  const auto AST = buildASTFromCode(
      "const int&& f() { int x; return static_cast<int&&>(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<int &&>(x)"));
}

// section: taking the address of a variable and pointers

TEST(ExprMutationAnalyzerTest, TakeAddress) {
  const auto AST = buildASTFromCode("void g(int*); void f() { int x; g(&x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("&x"));
}

TEST(ExprMutationAnalyzerTest, ArrayToPointerDecay) {
  const auto AST =
      buildASTFromCode("void g(int*); void f() { int x[2]; g(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x"));
}

TEST(ExprMutationAnalyzerTest, TemplateWithArrayToPointerDecay) {
  const auto AST = buildASTFromCodeWithArgs(
      "template <typename T> struct S { static constexpr int v = 8; };"
      "template <> struct S<int> { static constexpr int v = 4; };"
      "void g(char*);"
      "template <typename T> void f() { char x[S<T>::v]; g(x); }"
      "template <> void f<int>() { char y[S<int>::v]; g(y); }",
      {"-fno-delayed-template-parsing"});
  const auto ResultsX =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(ResultsX, AST.get()), ElementsAre("g(x)"));
  const auto ResultsY =
      match(withEnclosingCompound(declRefTo("y")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(ResultsY, AST.get()), ElementsAre("y"));
}

// section: special case: all created references are non-mutating themself
//          and therefore all become 'const'/the value is not modified!

TEST(ExprMutationAnalyzerTest, FollowRefModified) {
  auto AST = buildASTFromCode(
      "void f() { int x; int& r0 = x; int& r1 = r0; int& r2 = r1; "
      "int& r3 = r2; r3 = 10; }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("r0", "r1", "r2", "r3", "r3 = 10"));

  AST = buildASTFromCode("typedef int& IntRefX;"
                         "using IntRefY = int&;"
                         "void f() { int x; IntRefX r0 = x; IntRefY r1 = r0;"
                         "decltype((x)) r2 = r1; r2 = 10; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("r0", "r1", "r2", "r2 = 10"));
}

TEST(ExprMutationAnalyzerTest, FollowRefNotModified) {
  auto AST = buildASTFromCode(
      "void f() { int x; int& r0 = x; int& r1 = r0; int& r2 = r1; "
      "int& r3 = r2; int& r4 = r3; int& r5 = r4;}");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("void f() { int x; int& r0 = x; const int& r1 = r0;}");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("typedef const int& CIntRefX;"
                         "using CIntRefY = const int&;"
                         "void f() { int x; int& r0 = x; CIntRefX r1 = r0;"
                         "CIntRefY r2 = r1; decltype((r1)) r3 = r2;}");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, FollowConditionalRefModified) {
  const auto AST = buildASTFromCode(
      "void f() { int x, y; bool b; int &r = b ? x : y; r = 10; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("r", "r = 10"));
}

TEST(ExprMutationAnalyzerTest, FollowConditionalRefNotModified) {
  const auto AST =
      buildASTFromCode("void f() { int x, y; bool b; int& r = b ? x : y; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, FollowFuncArgModified) {
  auto AST = buildASTFromCode("template <class T> void g(T&& t) { t = 10; }"
                              "void f() { int x; g(x); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g(x)"));

  AST = buildASTFromCode(
      "void h(int&);"
      "template <class... Args> void g(Args&&... args) { h(args...); }"
      "void f() { int x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g(x)"));

  AST = buildASTFromCode(
      "void h(int&, int);"
      "template <class... Args> void g(Args&&... args) { h(args...); }"
      "void f() { int x, y; g(x, y); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g(x, y)"));
  Results = match(withEnclosingCompound(declRefTo("y")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(
      "void h(int, int&);"
      "template <class... Args> void g(Args&&... args) { h(args...); }"
      "void f() { int x, y; g(y, x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g(y, x)"));
  Results = match(withEnclosingCompound(declRefTo("y")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("struct S { template <class T> S(T&& t) { t = 10; } };"
                         "void f() { int x; S s(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x"));

  AST = buildASTFromCode(
      "struct S { template <class T> S(T&& t) : m(++t) { } int m; };"
      "void f() { int x; S s(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x"));

  AST = buildASTFromCode("template <class U> struct S {"
                         "template <class T> S(T&& t) : m(++t) { } U m; };"
                         "void f() { int x; S<int> s(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x"));

  AST = buildASTFromCode(StdRemoveReference + StdForward +
                         "template <class... Args> void u(Args&...);"
                         "template <class... Args> void h(Args&&... args)"
                         "{ u(std::forward<Args>(args)...); }"
                         "template <class... Args> void g(Args&&... args)"
                         "{ h(std::forward<Args>(args)...); }"
                         "void f() { int x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g(x)"));
}

TEST(ExprMutationAnalyzerTest, FollowFuncArgNotModified) {
  auto AST = buildASTFromCode("template <class T> void g(T&&) {}"
                              "void f() { int x; g(x); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("template <class T> void g(T&& t) { t; }"
                         "void f() { int x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("template <class... Args> void g(Args&&...) {}"
                         "void f() { int x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("template <class... Args> void g(Args&&...) {}"
                         "void f() { int y, x; g(y, x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(
      "void h(int, int&);"
      "template <class... Args> void g(Args&&... args) { h(args...); }"
      "void f() { int x, y; g(x, y); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("struct S { template <class T> S(T&& t) { t; } };"
                         "void f() { int x; S s(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(
      "struct S { template <class T> S(T&& t) : m(t) { } int m; };"
      "void f() { int x; S s(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("template <class U> struct S {"
                         "template <class T> S(T&& t) : m(t) { } U m; };"
                         "void f() { int x; S<int> s(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(StdRemoveReference + StdForward +
                         "template <class... Args> void u(Args...);"
                         "template <class... Args> void h(Args&&... args)"
                         "{ u(std::forward<Args>(args)...); }"
                         "template <class... Args> void g(Args&&... args)"
                         "{ h(std::forward<Args>(args)...); }"
                         "void f() { int x; g(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

// section: builtin arrays

TEST(ExprMutationAnalyzerTest, ArrayElementModified) {
  const auto AST = buildASTFromCode("void f() { int x[2]; x[0] = 10; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x[0] = 10"));
}

TEST(ExprMutationAnalyzerTest, ArrayElementNotModified) {
  const auto AST = buildASTFromCode("void f() { int x[2]; x[0]; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

// section: member modifications

TEST(ExprMutationAnalyzerTest, NestedMemberModified) {
  auto AST =
      buildASTFromCode("void f() { struct A { int vi; }; struct B { A va; }; "
                       "struct C { B vb; }; C x; x.vb.va.vi = 10; }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x.vb.va.vi = 10"));

  AST = buildASTFromCodeWithArgs(
      "template <class T> void f() { T x; x.y.z = 10; }",
      {"-fno-delayed-template-parsing"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x.y.z = 10"));

  AST = buildASTFromCodeWithArgs(
      "template <class T> struct S;"
      "template <class T> void f() { S<T> x; x.y.z = 10; }",
      {"-fno-delayed-template-parsing"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x.y.z = 10"));
}

TEST(ExprMutationAnalyzerTest, NestedMemberNotModified) {
  auto AST =
      buildASTFromCode("void f() { struct A { int vi; }; struct B { A va; }; "
                       "struct C { B vb; }; C x; x.vb.va.vi; }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCodeWithArgs("template <class T> void f() { T x; x.y.z; }",
                                 {"-fno-delayed-template-parsing"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST =
      buildASTFromCodeWithArgs("template <class T> struct S;"
                               "template <class T> void f() { S<T> x; x.y.z; }",
                               {"-fno-delayed-template-parsing"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

// section: casts

TEST(ExprMutationAnalyzerTest, CastToValue) {
  const auto AST =
      buildASTFromCode("void f() { int x; static_cast<double>(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CastToRefModified) {
  auto AST =
      buildASTFromCode("void f() { int x; static_cast<int &>(x) = 10; }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<int &>(x)"));

  AST = buildASTFromCode("typedef int& IntRef;"
                         "void f() { int x; static_cast<IntRef>(x) = 10; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<IntRef>(x)"));
}

TEST(ExprMutationAnalyzerTest, CastToRefNotModified) {
  const auto AST =
      buildASTFromCode("void f() { int x; static_cast<int&>(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<int &>(x)"));
}

TEST(ExprMutationAnalyzerTest, CastToConstRef) {
  auto AST =
      buildASTFromCode("void f() { int x; static_cast<const int&>(x); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("typedef const int& CIntRef;"
                         "void f() { int x; static_cast<CIntRef>(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

// section: comma expressions

TEST(ExprMutationAnalyzerTest, CommaExprWithAnAssigment) {
  const auto AST = buildASTFromCodeWithArgs(
      "void f() { int x; int y; (x, y) = 5; }", {"-Wno-unused-value"});
  const auto Results =
      match(withEnclosingCompound(declRefTo("y")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CommaExprWithDecOp) {
  const auto AST = buildASTFromCodeWithArgs(
      "void f() { int x; int y; (x, y)++; }", {"-Wno-unused-value"});
  const auto Results =
      match(withEnclosingCompound(declRefTo("y")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CommaExprWithNonConstMemberCall) {
  const auto AST = buildASTFromCodeWithArgs(
      "class A { public: int mem; void f() { mem ++; } };"
      "void fn() { A o1, o2; (o1, o2).f(); }",
      {"-Wno-unused-value"});
  const auto Results =
      match(withEnclosingCompound(declRefTo("o2")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CommaExprWithConstMemberCall) {
  const auto AST = buildASTFromCodeWithArgs(
      "class A { public: int mem; void f() const  { } };"
      "void fn() { A o1, o2; (o1, o2).f(); }",
      {"-Wno-unused-value"});
  const auto Results =
      match(withEnclosingCompound(declRefTo("o2")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CommaExprWithCallExpr) {
  const auto AST =
      buildASTFromCodeWithArgs("class A { public: int mem; void f(A &O1) {} };"
                               "void fn() { A o1, o2; o2.f((o2, o1)); }",
                               {"-Wno-unused-value"});
  const auto Results =
      match(withEnclosingCompound(declRefTo("o1")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CommaExprWithCallUnresolved) {
  auto AST = buildASTFromCodeWithArgs(
      "template <class T> struct S;"
      "template <class T> void f() { S<T> s; int x, y; s.mf((y, x)); }",
      {"-fno-delayed-template-parsing", "-Wno-unused-value"});
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));

  AST = buildASTFromCodeWithArgs(
      "template <class T> void f(T t) { int x, y; g(t, (y, x)); }",
      {"-fno-delayed-template-parsing", "-Wno-unused-value"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CommaExprParmRef) {
  const auto AST =
      buildASTFromCodeWithArgs("class A { public: int mem;};"
                               "extern void fn(A &o1);"
                               "void fn2 () { A o1, o2; fn((o2, o1)); } ",
                               {"-Wno-unused-value"});
  const auto Results =
      match(withEnclosingCompound(declRefTo("o1")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CommaExprWithAmpersandOp) {
  const auto AST = buildASTFromCodeWithArgs("class A { public: int mem;};"
                                            "void fn () { A o1, o2;"
                                            "void *addr = &(o2, o1); } ",
                                            {"-Wno-unused-value"});
  const auto Results =
      match(withEnclosingCompound(declRefTo("o1")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CommaExprAsReturnAsValue) {
  auto AST = buildASTFromCodeWithArgs("int f() { int x, y; return (x, y); }",
                                      {"-Wno-unused-value"});
  auto Results =
      match(withEnclosingCompound(declRefTo("y")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CommaEpxrAsReturnAsNonConstRef) {
  const auto AST = buildASTFromCodeWithArgs(
      "int& f() { int x, y; return (y, x); }", {"-Wno-unused-value"});
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CommaExprAsArrayToPointerDecay) {
  const auto AST =
      buildASTFromCodeWithArgs("void g(int*); "
                               "void f() { int x[2], y[2]; g((y, x)); }",
                               {"-Wno-unused-value"});
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CommaExprAsUniquePtr) {
  const std::string UniquePtrDef = "template <class T> struct UniquePtr {"
                                   "  UniquePtr();"
                                   "  UniquePtr(const UniquePtr&) = delete;"
                                   "  T& operator*() const;"
                                   "  T* operator->() const;"
                                   "};";
  const auto AST = buildASTFromCodeWithArgs(
      UniquePtrDef + "template <class T> void f() "
                     "{ UniquePtr<T> x; UniquePtr<T> y;"
                     " (y, x)->mf(); }",
      {"-fno-delayed-template-parsing", "-Wno-unused-value"});
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CommaNestedConditional) {
  const std::string Code = "void f() { int x, y = 42;"
                           " y, (true ? x : y) = 42; }";
  const auto AST = buildASTFromCodeWithArgs(Code, {"-Wno-unused-value"});
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("(true ? x : y) = 42"));
}

// section: lambda captures

TEST(ExprMutationAnalyzerTest, LambdaDefaultCaptureByValue) {
  const auto AST = buildASTFromCode("void f() { int x; [=]() { x; }; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, LambdaExplicitCaptureByValue) {
  const auto AST = buildASTFromCode("void f() { int x; [x]() { x; }; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, LambdaDefaultCaptureByRef) {
  const auto AST = buildASTFromCode("void f() { int x; [&]() { x = 10; }; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre(ResultOf(removeSpace, "[&](){x=10;}")));
}

TEST(ExprMutationAnalyzerTest, LambdaExplicitCaptureByRef) {
  const auto AST = buildASTFromCode("void f() { int x; [&x]() { x = 10; }; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre(ResultOf(removeSpace, "[&x](){x=10;}")));
}

// section: range-for loops

TEST(ExprMutationAnalyzerTest, RangeForArrayByRefModified) {
  auto AST =
      buildASTFromCode("void f() { int x[2]; for (int& e : x) e = 10; }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("for (int &e : x)\n    e = 10;"));

  AST = buildASTFromCode("typedef int& IntRef;"
                         "void f() { int x[2]; for (IntRef e : x) e = 10; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("for (IntRef e : x)\n    e = 10;"));
}

TEST(ExprMutationAnalyzerTest, RangeForArrayByRefModifiedByImplicitInit) {
  const auto AST =
      buildASTFromCode("void f() { int x[2]; for (int& e : x) e; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, RangeForArrayByValue) {
  auto AST = buildASTFromCode("void f() { int x[2]; for (int e : x) e = 10; }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST =
      buildASTFromCode("void f() { int* x[2]; for (int* e : x) e = nullptr; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(
      "typedef int* IntPtr;"
      "void f() { int* x[2]; for (IntPtr e : x) e = nullptr; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, RangeForArrayByConstRef) {
  auto AST =
      buildASTFromCode("void f() { int x[2]; for (const int& e : x) e; }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("typedef const int& CIntRef;"
                         "void f() { int x[2]; for (CIntRef e : x) e; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, RangeForNonArrayByRefModified) {
  const auto AST =
      buildASTFromCode("struct V { int* begin(); int* end(); };"
                       "void f() { V x; for (int& e : x) e = 10; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("for (int &e : x)\n    e = 10;"));
}

TEST(ExprMutationAnalyzerTest, RangeForNonArrayByRefNotModified) {
  const auto AST = buildASTFromCode("struct V { int* begin(); int* end(); };"
                                    "void f() { V x; for (int& e : x) e; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, RangeForNonArrayByValue) {
  const auto AST = buildASTFromCode(
      "struct V { const int* begin() const; const int* end() const; };"
      "void f() { V x; for (int e : x) e; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, RangeForNonArrayByConstRef) {
  const auto AST = buildASTFromCode(
      "struct V { const int* begin() const; const int* end() const; };"
      "void f() { V x; for (const int& e : x) e; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

// section: unevaluated expressions

TEST(ExprMutationAnalyzerTest, UnevaluatedExpressions) {
  auto AST = buildASTFromCode("void f() { int x, y; decltype(x = 10) z = y; }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("void f() { int x, y; __typeof(x = 10) z = y; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("void f() { int x, y; __typeof__(x = 10) z = y; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("void f() { int x; sizeof(x = 10); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("void f() { int x; alignof(x = 10); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("void f() { int x; noexcept(x = 10); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCodeWithArgs("namespace std { class type_info; }"
                                 "void f() { int x; typeid(x = 10); }",
                                 {"-frtti"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(
      "void f() { int x; _Generic(x = 10, int: 0, default: 1); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, NotUnevaluatedExpressions) {
  auto AST = buildASTFromCode("void f() { int x; sizeof(int[x++]); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x++"));

  AST = buildASTFromCodeWithArgs(
      "namespace std { class type_info; }"
      "struct A { virtual ~A(); }; struct B : A {};"
      "struct X { A& f(); }; void f() { X x; typeid(x.f()); }",
      {"-frtti"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x.f()"));
}

// section: special case: smartpointers

TEST(ExprMutationAnalyzerTest, UniquePtr) {
  const std::string UniquePtrDef =
      "template <class T> struct UniquePtr {"
      "  UniquePtr();"
      "  UniquePtr(const UniquePtr&) = delete;"
      "  UniquePtr(UniquePtr&&);"
      "  UniquePtr& operator=(const UniquePtr&) = delete;"
      "  UniquePtr& operator=(UniquePtr&&);"
      "  T& operator*() const;"
      "  T* operator->() const;"
      "};";

  auto AST = buildASTFromCode(UniquePtrDef +
                              "void f() { UniquePtr<int> x; *x = 10; }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("* x = 10"));

  AST = buildASTFromCode(UniquePtrDef + "void f() { UniquePtr<int> x; *x; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(UniquePtrDef +
                         "void f() { UniquePtr<const int> x; *x; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(UniquePtrDef + "struct S { int v; };"
                                        "void f() { UniquePtr<S> x; x->v; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x"));

  AST = buildASTFromCode(UniquePtrDef +
                         "struct S { int v; };"
                         "void f() { UniquePtr<const S> x; x->v; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST =
      buildASTFromCode(UniquePtrDef + "struct S { void mf(); };"
                                      "void f() { UniquePtr<S> x; x->mf(); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x"));

  AST = buildASTFromCode(UniquePtrDef +
                         "struct S { void mf() const; };"
                         "void f() { UniquePtr<const S> x; x->mf(); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCodeWithArgs(
      UniquePtrDef + "template <class T> void f() { UniquePtr<T> x; x->mf(); }",
      {"-fno-delayed-template-parsing"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x->mf()"));
}

// section: complex problems detected on real code

TEST(ExprMutationAnalyzerTest, UnevaluatedContext) {
  const std::string Example =
      "template <typename T>"
      "struct to_construct : T { to_construct(int &j) {} };"
      "template <typename T>"
      "void placement_new_in_unique_ptr() { int x = 0;"
      "  new to_construct<T>(x);"
      "}";
  auto AST =
      buildASTFromCodeWithArgs(Example, {"-fno-delayed-template-parsing"});
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_TRUE(isMutated(Results, AST.get()));
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("(x)"));
}

TEST(ExprMutationAnalyzerTest, ReproduceFailureMinimal) {
  const std::string Reproducer =
      "namespace std {"
      "template <class T> T &forward(T &A) { return static_cast<T&&>(A); }"
      "template <class T> struct __bind {"
      "  T f;"
      "  template <class V> __bind(T v, V &&) : f(forward(v)) {}"
      "};"
      "}"
      "void f() {"
      "  int x = 42;"
      "  auto Lambda = [] {};"
      "  std::__bind<decltype(Lambda)>(Lambda, x);"
      "}";
  auto AST11 = buildASTFromCodeWithArgs(Reproducer, {"-std=c++11"});
  auto Results11 =
      match(withEnclosingCompound(declRefTo("x")), AST11->getASTContext());
  EXPECT_FALSE(isMutated(Results11, AST11.get()));
}
} // namespace clang
