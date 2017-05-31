//===- unittests/AST/DeclTest.cpp --- Declaration tests -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Decl nodes in the AST.
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang::ast_matchers;
using namespace clang::tooling;

TEST(Decl, CleansUpAPValues) {
  MatchFinder Finder;
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));

  // This is a regression test for a memory leak in APValues for structs that
  // allocate memory. This test only fails if run under valgrind with full leak
  // checking enabled.
  std::vector<std::string> Args(1, "-std=c++11");
  Args.push_back("-fno-ms-extensions");
  ASSERT_TRUE(runToolOnCodeWithArgs(
      Factory->create(),
      "struct X { int a; }; constexpr X x = { 42 };"
      "union Y { constexpr Y(int a) : a(a) {} int a; }; constexpr Y y = { 42 };"
      "constexpr int z[2] = { 42, 43 };"
      "constexpr int __attribute__((vector_size(16))) v1 = {};"
      "\n#ifdef __SIZEOF_INT128__\n"
      "constexpr __uint128_t large_int = 0xffffffffffffffff;"
      "constexpr __uint128_t small_int = 1;"
      "\n#endif\n"
      "constexpr double d1 = 42.42;"
      "constexpr long double d2 = 42.42;"
      "constexpr _Complex long double c1 = 42.0i;"
      "constexpr _Complex long double c2 = 42.0;"
      "template<int N> struct A : A<N-1> {};"
      "template<> struct A<0> { int n; }; A<50> a;"
      "constexpr int &r = a.n;"
      "constexpr int A<50>::*p = &A<50>::n;"
      "void f() { foo: bar: constexpr int k = __builtin_constant_p(0) ?"
      "                         (char*)&&foo - (char*)&&bar : 0; }",
      Args));

  // FIXME: Once this test starts breaking we can test APValue::needsCleanup
  // for ComplexInt.
  ASSERT_FALSE(runToolOnCodeWithArgs(
      Factory->create(),
      "constexpr _Complex __uint128_t c = 0xffffffffffffffff;",
      Args));
}
