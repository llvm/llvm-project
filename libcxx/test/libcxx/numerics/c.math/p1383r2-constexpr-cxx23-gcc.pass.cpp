//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that GCC supports constexpr <cmath> and <cstdlib> functions
// mentioned in the P1383R2 paper that is part of C++23
// (https://wg21.link/p1383r2)
//
// Every function called in this test should become constexpr. Whenever some
// of the desired function become constexpr, the programmer switches
// `ASSERT_NOT_CONSTEXPR_CXX23` to `ASSERT_CONSTEXPR_CXX23` and eventually the
// paper is implemented in libc++.
// The test also works as a reference list of unimplemented functions.
//
// REQUIRES: gcc 
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cassert>
#include <cmath>
#include <cstdlib>

int main(int, char**) {
  bool ImplementedP1383R2 = true;

#define ASSERT_CONSTEXPR_CXX23(Expr) static_assert(__builtin_constant_p(Expr) && (Expr))
#define ASSERT_NOT_CONSTEXPR_CXX23(Expr)                                                                               \
  static_assert(!__builtin_constant_p(Expr));                                                                          \
  assert(Expr);                                                                                                        \
  ImplementedP1383R2 = false

  int DummyInt;
  float DummyFloat;
  double DummyDouble;
  long double DummyLongDouble;

  assert(!ImplementedP1383R2 && R"(
Congratulations! You just have implemented P1383R2 (https://wg21.link/p1383r2).
Please go to `clang/www/cxx_status.html` and change the paper's implementation
status. Also please delete this assert and refactor `ASSERT_CONSTEXPR_CXX23`
and `ASSERT_NOT_CONSTEXPR_CXX23`.
)");

  return 0;
}
