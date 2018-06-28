//===---------- ExprMutationAnalyzerTest.cpp - clang-tidy -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../clang-tidy/utils/ExprMutationAnalyzer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cctype>

namespace clang {
namespace tidy {
namespace test {

using namespace clang::ast_matchers;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::ResultOf;
using ::testing::StartsWith;
using ::testing::Values;

namespace {

using ExprMatcher = internal::Matcher<Expr>;
using StmtMatcher = internal::Matcher<Stmt>;

ExprMatcher declRefTo(StringRef Name) {
  return declRefExpr(to(namedDecl(hasName(Name))));
}

StmtMatcher withEnclosingCompound(ExprMatcher Matcher) {
  return expr(Matcher, hasAncestor(compoundStmt().bind("stmt"))).bind("expr");
}

bool isMutated(const SmallVectorImpl<BoundNodes> &Results, ASTUnit *AST) {
  const auto *const S = selectFirst<Stmt>("stmt", Results);
  const auto *const E = selectFirst<Expr>("expr", Results);
  return utils::ExprMutationAnalyzer(S, &AST->getASTContext()).isMutated(E);
}

SmallVector<std::string, 1>
mutatedBy(const SmallVectorImpl<BoundNodes> &Results, ASTUnit *AST) {
  const auto *const S = selectFirst<Stmt>("stmt", Results);
  SmallVector<std::string, 1> Chain;
  utils::ExprMutationAnalyzer Analyzer(S, &AST->getASTContext());
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

} // namespace

TEST(ExprMutationAnalyzerTest, Trivial) {
  const auto AST = tooling::buildASTFromCode("void f() { int x; x; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

class AssignmentTest : public ::testing::TestWithParam<std::string> {};

TEST_P(AssignmentTest, AssignmentModifies) {
  const std::string ModExpr = "x " + GetParam() + " 10";
  const auto AST =
      tooling::buildASTFromCode("void f() { int x; " + ModExpr + "; }");
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
  const auto AST =
      tooling::buildASTFromCode("void f() { int x; " + ModExpr + "; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre(ModExpr));
}

INSTANTIATE_TEST_CASE_P(AllIncDecOperators, IncDecTest,
                        Values("++x", "--x", "x++", "x--"), );

TEST(ExprMutationAnalyzerTest, NonConstMemberFunc) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { struct Foo { void mf(); }; Foo x; x.mf(); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x.mf()"));
}

TEST(ExprMutationAnalyzerTest, ConstMemberFunc) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { struct Foo { void mf() const; }; Foo x; x.mf(); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, NonConstOperator) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { struct Foo { Foo& operator=(int); }; Foo x; x = 10; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x = 10"));
}

TEST(ExprMutationAnalyzerTest, ConstOperator) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { struct Foo { int operator()() const; }; Foo x; x(); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, ByValueArgument) {
  auto AST =
      tooling::buildASTFromCode("void g(int); void f() { int x; g(x); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode(
      "void f() { struct A {}; A operator+(A, int); A x; x + 1; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode(
      "void f() { struct A { A(int); }; int x; A y(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode(
      "void f() { struct A { A(); A(A); }; A x; A y(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, ByNonConstRefArgument) {
  auto AST =
      tooling::buildASTFromCode("void g(int&); void f() { int x; g(x); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("g(x)"));

  AST = tooling::buildASTFromCode(
      "void f() { struct A {}; A operator+(A&, int); A x; x + 1; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x + 1"));

  AST = tooling::buildASTFromCode(
      "void f() { struct A { A(int&); }; int x; A y(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x"));

  AST = tooling::buildASTFromCode(
      "void f() { struct A { A(); A(A&); }; A x; A y(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x"));
}

TEST(ExprMutationAnalyzerTest, ByConstRefArgument) {
  auto AST = tooling::buildASTFromCode(
      "void g(const int&); void f() { int x; g(x); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode(
      "void f() { struct A {}; A operator+(const A&, int); A x; x + 1; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode(
      "void f() { struct A { A(const int&); }; int x; A y(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode(
      "void f() { struct A { A(); A(const A&); }; A x; A y(x); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, ByNonConstRRefArgument) {
  auto AST = tooling::buildASTFromCode(
      "void g(int&&); void f() { int x; g(static_cast<int &&>(x)); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("g(static_cast<int &&>(x))"));

  AST = tooling::buildASTFromCode(
      "void f() { struct A {}; A operator+(A&&, int); "
      "A x; static_cast<A &&>(x) + 1; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<A &&>(x) + 1"));

  AST = tooling::buildASTFromCode("void f() { struct A { A(int&&); }; "
                                  "int x; A y(static_cast<int &&>(x)); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<int &&>(x)"));

  AST = tooling::buildASTFromCode("void f() { struct A { A(); A(A&&); }; "
                                  "A x; A y(static_cast<A &&>(x)); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<A &&>(x)"));
}

TEST(ExprMutationAnalyzerTest, ByConstRRefArgument) {
  auto AST = tooling::buildASTFromCode(
      "void g(const int&&); void f() { int x; g(static_cast<int&&>(x)); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode(
      "void f() { struct A {}; A operator+(const A&&, int); "
      "A x; static_cast<A&&>(x) + 1; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode("void f() { struct A { A(const int&&); }; "
                                  "int x; A y(static_cast<int&&>(x)); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode("void f() { struct A { A(); A(const A&&); }; "
                                  "A x; A y(static_cast<A&&>(x)); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, Move) {
  // Technically almost the same as ByNonConstRRefArgument, just double checking
  const auto AST = tooling::buildASTFromCode(
      "namespace std {"
      "template<class T> struct remove_reference { typedef T type; };"
      "template<class T> struct remove_reference<T&> { typedef T type; };"
      "template<class T> struct remove_reference<T&&> { typedef T type; };"
      "template<class T> typename std::remove_reference<T>::type&& "
      "move(T&& t) noexcept; }"
      "void f() { struct A {}; A x; std::move(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("std::move(x)"));
}

TEST(ExprMutationAnalyzerTest, Forward) {
  // Technically almost the same as ByNonConstRefArgument, just double checking
  const auto AST = tooling::buildASTFromCode(
      "namespace std {"
      "template<class T> struct remove_reference { typedef T type; };"
      "template<class T> struct remove_reference<T&> { typedef T type; };"
      "template<class T> struct remove_reference<T&&> { typedef T type; };"
      "template<class T> T&& "
      "forward(typename std::remove_reference<T>::type&) noexcept;"
      "template<class T> T&& "
      "forward(typename std::remove_reference<T>::type&&) noexcept;"
      "void f() { struct A {}; A x; std::forward<A &>(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("std::forward<A &>(x)"));
}

TEST(ExprMutationAnalyzerTest, ReturnAsValue) {
  const auto AST = tooling::buildASTFromCode("int f() { int x; return x; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, ReturnAsNonConstRef) {
  const auto AST = tooling::buildASTFromCode("int& f() { int x; return x; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("return x;"));
}

TEST(ExprMutationAnalyzerTest, ReturnAsConstRef) {
  const auto AST =
      tooling::buildASTFromCode("const int& f() { int x; return x; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, ReturnAsNonConstRRef) {
  const auto AST = tooling::buildASTFromCode(
      "int&& f() { int x; return static_cast<int &&>(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("return static_cast<int &&>(x);"));
}

TEST(ExprMutationAnalyzerTest, ReturnAsConstRRef) {
  const auto AST = tooling::buildASTFromCode(
      "const int&& f() { int x; return static_cast<int&&>(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, TakeAddress) {
  const auto AST =
      tooling::buildASTFromCode("void g(int*); void f() { int x; g(&x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("&x"));
}

TEST(ExprMutationAnalyzerTest, ArrayToPointerDecay) {
  const auto AST =
      tooling::buildASTFromCode("void g(int*); void f() { int x[2]; g(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x"));
}

TEST(ExprMutationAnalyzerTest, FollowRefModified) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { int x; int& r0 = x; int& r1 = r0; int& r2 = r1; "
      "int& r3 = r2; r3 = 10; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("r0", "r1", "r2", "r3", "r3 = 10"));
}

TEST(ExprMutationAnalyzerTest, FollowRefNotModified) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { int x; int& r0 = x; int& r1 = r0; int& r2 = r1; "
      "int& r3 = r2; int& r4 = r3; int& r5 = r4;}");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, FollowConditionalRefModified) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { int x, y; bool b; int &r = b ? x : y; r = 10; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("r", "r = 10"));
}

TEST(ExprMutationAnalyzerTest, FollowConditionalRefNotModified) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { int x, y; bool b; int& r = b ? x : y; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, ArrayElementModified) {
  const auto AST =
      tooling::buildASTFromCode("void f() { int x[2]; x[0] = 10; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x[0] = 10"));
}

TEST(ExprMutationAnalyzerTest, ArrayElementNotModified) {
  const auto AST = tooling::buildASTFromCode("void f() { int x[2]; x[0]; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, NestedMemberModified) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { struct A { int vi; }; struct B { A va; }; "
      "struct C { B vb; }; C x; x.vb.va.vi = 10; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x.vb.va.vi = 10"));
}

TEST(ExprMutationAnalyzerTest, NestedMemberNotModified) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { struct A { int vi; }; struct B { A va; }; "
      "struct C { B vb; }; C x; x.vb.va.vi; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CastToValue) {
  const auto AST =
      tooling::buildASTFromCode("void f() { int x; static_cast<double>(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CastToRefModified) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { int x; static_cast<int &>(x) = 10; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre("static_cast<int &>(x) = 10"));
}

TEST(ExprMutationAnalyzerTest, CastToRefNotModified) {
  const auto AST =
      tooling::buildASTFromCode("void f() { int x; static_cast<int&>(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, CastToConstRef) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { int x; static_cast<const int&>(x); }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, LambdaDefaultCaptureByValue) {
  const auto AST =
      tooling::buildASTFromCode("void f() { int x; [=]() { x = 10; }; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, LambdaExplicitCaptureByValue) {
  const auto AST =
      tooling::buildASTFromCode("void f() { int x; [x]() { x = 10; }; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, LambdaDefaultCaptureByRef) {
  const auto AST =
      tooling::buildASTFromCode("void f() { int x; [&]() { x = 10; }; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre(ResultOf(removeSpace, "[&](){x=10;}")));
}

TEST(ExprMutationAnalyzerTest, LambdaExplicitCaptureByRef) {
  const auto AST =
      tooling::buildASTFromCode("void f() { int x; [&x]() { x = 10; }; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()),
              ElementsAre(ResultOf(removeSpace, "[&x](){x=10;}")));
}

TEST(ExprMutationAnalyzerTest, RangeForArrayByRefModified) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { int x[2]; for (int& e : x) e = 10; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("e", "e = 10"));
}

TEST(ExprMutationAnalyzerTest, RangeForArrayByRefNotModified) {
  const auto AST =
      tooling::buildASTFromCode("void f() { int x[2]; for (int& e : x) e; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, RangeForArrayByValue) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { int x[2]; for (int e : x) e = 10; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, RangeForArrayByConstRef) {
  const auto AST = tooling::buildASTFromCode(
      "void f() { int x[2]; for (const int& e : x) e; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, RangeForNonArrayByRefModified) {
  const auto AST =
      tooling::buildASTFromCode("struct V { int* begin(); int* end(); };"
                                "void f() { V x; for (int& e : x) e = 10; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("e", "e = 10"));
}

TEST(ExprMutationAnalyzerTest, RangeForNonArrayByRefNotModified) {
  const auto AST =
      tooling::buildASTFromCode("struct V { int* begin(); int* end(); };"
                                "void f() { V x; for (int& e : x) e; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, RangeForNonArrayByValue) {
  const auto AST = tooling::buildASTFromCode(
      "struct V { const int* begin() const; const int* end() const; };"
      "void f() { V x; for (int e : x) e; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, RangeForNonArrayByConstRef) {
  const auto AST = tooling::buildASTFromCode(
      "struct V { const int* begin() const; const int* end() const; };"
      "void f() { V x; for (const int& e : x) e; }");
  const auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, UnevaluatedExpressions) {
  auto AST = tooling::buildASTFromCode(
      "void f() { int x, y; decltype(x = 10) z = y; }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode(
      "void f() { int x, y; __typeof(x = 10) z = y; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode(
      "void f() { int x, y; __typeof__(x = 10) z = y; }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode("void f() { int x; sizeof(x = 10); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode("void f() { int x; alignof(x = 10); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode("void f() { int x; noexcept(x = 10); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCodeWithArgs("namespace std { class type_info; }"
                                          "void f() { int x; typeid(x = 10); }",
                                          {"-frtti"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));

  AST = tooling::buildASTFromCode(
      "void f() { int x; _Generic(x = 10, int: 0, default: 1); }");
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_FALSE(isMutated(Results, AST.get()));
}

TEST(ExprMutationAnalyzerTest, NotUnevaluatedExpressions) {
  auto AST = tooling::buildASTFromCode("void f() { int x; sizeof(int[x++]); }");
  auto Results =
      match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x++"));

  AST = tooling::buildASTFromCodeWithArgs(
      "namespace std { class type_info; }"
      "struct A { virtual ~A(); }; struct B : A {};"
      "struct X { A& f(); }; void f() { X x; typeid(x.f()); }",
      {"-frtti"});
  Results = match(withEnclosingCompound(declRefTo("x")), AST->getASTContext());
  EXPECT_THAT(mutatedBy(Results, AST.get()), ElementsAre("x.f()"));
}

} // namespace test
} // namespace tidy
} // namespace clang
