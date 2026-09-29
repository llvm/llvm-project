//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// REQUIRES: gcc

// Check that GCC supports constexpr <cmath> and <cstdlib> functions
// mentioned in the P1383R2 paper that is part of C++26
// (https://wg21.link/P1383R2)
//
// Every function called in this test should become constexpr. Whenever some
// of the desired function become constexpr, the programmer switches
// `ASSERT_NOT_CONSTEXPR_CXX26` to `ASSERT_CONSTEXPR_CXX26` and eventually the
// paper is implemented in Clang.
// The test also works as a reference list of unimplemented functions.

#include <cassert>
#include <cmath>
#include <cstdlib>

int main(int, char**) {
  bool ImplementedP1383R2 = true;

#define ASSERT_CONSTEXPR_CXX26(Expr) static_assert(__builtin_constant_p(Expr) && (Expr))
#define ASSERT_NOT_CONSTEXPR_CXX26(Expr)                                                                               \
  static_assert(!__builtin_constant_p(Expr));                                                                          \
  assert(Expr);                                                                                                        \
  ImplementedP1383R2 = false

  // acos
  ASSERT_NOT_CONSTEXPR_CXX26(std::acos(1.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::acos(1.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::acos(1.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::acosf(1.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::acosl(1.0L) == 0.0L);

  // asin()
  ASSERT_NOT_CONSTEXPR_CXX26(std::asin(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::asin(0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::asin(0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::asinf(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::asinl(0.0L) == 0.0L);

  // atan()
  ASSERT_NOT_CONSTEXPR_CXX26(std::atan(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::atan(0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::atan(0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::atanf(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::atanl(0.0L) == 0.0L);

  assert(!ImplementedP1383R2 && R"(
Congratulations! You just have implemented P1383R2 (https://wg21.link/P1383R2).
Please go to `clang/www/cxx_status.html` and change the paper's implementation
status. Also please delete this assert and refactor `ASSERT_CONSTEXPR_CXX26`
and `ASSERT_NOT_CONSTEXPR_CXX26`.
)");

  return 0;
}
