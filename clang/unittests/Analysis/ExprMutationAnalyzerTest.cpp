//===---------- ExprMutationAnalyzerTest.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"
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
using ::testing::IsEmpty;
using ::testing::ResultOf;
using ::testing::StartsWith;
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
  return ExprMutationAnalyzer(*S, AST->getASTContext()).isMutated(E);
}

SmallVector<std::string, 1>
mutatedBy(const SmallVectorImpl<BoundNodes> &Results, ASTUnit *AST) {
  const auto *const S = selectFirst<Stmt>("stmt", Results);
  SmallVector<std::string, 1> Chain;
  ExprMutationAnalyzer Analyzer(*S, AST->getASTContext());
  for (const auto *E = selectFirst<Expr>("expr", Results); E != nullptr;) {
    const Stmt *By = Analyzer.findMutation(E);
    std::string buffer;
    llvm::raw_string_ostream stream(buffer);
    By->printPretty(stream, nullptr, AST->getASTContext().getPrintingPolicy());
    Chain.push_back(StringRef(stream.str()).trim().str());
    E = dyn_cast<DeclRefExpr>(By);
  }
  return Chain;
}

std::string removeSpace(std::string s) {
  s.erase(std::remove_if(s.begin(), s.end(),
                         [](char c) { return std::isspace(c); }),
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

TEST_P(AssignmentTest, AssignmentModifies) {
  const std::string ModExpr = "x " + GetParam() + " 10";
  const auto AST = buildASTFromCode("void f() { int x; " + ModExpr + "; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre(ModExpr));
}

INSTANTIATE_TEST_CASE_P(AllAssignmentOperators, AssignmentTest,
                        Values("=", "+=", "-=", "*=", "/=", "%=", "&=", "|=",
                               "^=", "<<=", ">>="), );

class IncDecTest : public ::testing::TestWithParam<std::string> {};

TEST_P(IncDecTest, IncDecModifies) {
  const std::string ModExpr = GetParam();
  const auto AST = buildASTFromCode("void f() { int x; " + ModExpr + "; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre(ModExpr));
}

INSTANTIATE_TEST_CASE_P(AllIncDecOperators, IncDecTest,
                        Values("++x", "--x", "x++", "x--"), );

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
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("struct A {}; A operator+(const A&&, int);"
                         "void f() { A x; static_cast<A&&>(x) + 1; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("void f() { struct A { A(const int&&); }; "
                         "int x; A y(static_cast<int&&>(x)); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode("void f() { struct A { A(); A(const A&&); }; "
                         "A x; A y(static_cast<A&&>(x)); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

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
              ElementsAre("return static_cast<int &&>(x);"));
}

TEST(ExprMutationAnalyzerTest, ReturnAsConstRRef) {
  const auto AST = buildASTFromCode(
      "const int&& f() { int x; return static_cast<int&&>(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

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
              ElementsAre("static_cast<int &>(x) = 10"));

  AST = buildASTFromCode("typedef int& IntRef;"
                         "void f() { int x; static_cast<IntRef>(x) = 10; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<IntRef>(x) = 10"));
}

TEST(ExprMutationAnalyzerTest, CastToRefNotModified) {
  const auto AST =
      buildASTFromCode("void f() { int x; static_cast<int&>(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
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

TEST(ExprMutationAnalyzerTest, RangeForArrayByRefModified) {
  auto AST =
      buildASTFromCode("void f() { int x[2]; for (int& e : x) e = 10; }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("e", "e = 10"));

  AST = buildASTFromCode("typedef int& IntRef;"
                         "void f() { int x[2]; for (IntRef e : x) e = 10; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("e", "e = 10"));
}

TEST(ExprMutationAnalyzerTest, RangeForArrayByRefNotModified) {
  const auto AST =
      buildASTFromCode("void f() { int x[2]; for (int& e : x) e; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, RangeForArrayByValue) {
  auto AST = buildASTFromCode("void f() { int x[2]; for (int e : x) e = 10; }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST =
      buildASTFromCode("void f() { int* x[2]; for (int* e : x) e = nullptr; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = buildASTFromCode(
      "typedef int* IntPtr;"
      "void f() { int* x[2]; for (IntPtr e : x) e = nullptr; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
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
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("e", "e = 10"));
}

TEST(ExprMutationAnalyzerTest, RangeForNonArrayByRefNotModified) {
  const auto AST = buildASTFromCode("struct V { int* begin(); int* end(); };"
                                    "void f() { V x; for (int& e : x) e; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
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

} // namespace clang
