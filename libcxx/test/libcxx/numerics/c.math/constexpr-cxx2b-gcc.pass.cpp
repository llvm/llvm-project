//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that GCC supports constexpr <cmath> and <cstdlib> functions
// mentioned in the P0533R9 paper that is part of C++2b
// (https://wg21.link/p0533r9)
//
// Every function called in this test should become constexpr. Whenever some
// of the desired function become constexpr, the programmer switches
// `ASSERT_NOT_CONSTEXPR_CXX23` to `ASSERT_CONSTEXPR_CXX23` and eventually the
// paper is implemented in Clang.
// The test also works as a reference list of unimplemented functions.
//
// REQUIRES: gcc
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cmath>
#include <cstdlib>
#include <cassert>

int main(int, char**) {
  bool ImplementedP0533R9 = true;

#define ASSERT_CONSTEXPR_CXX23(Expr) static_assert(__builtin_constant_p(Expr) && (Expr))
#define ASSERT_NOT_CONSTEXPR_CXX23(Expr)                                                                               \
  static_assert(!__builtin_constant_p(Expr));                                                                          \
  assert(Expr);                                                                                                        \
  ImplementedP0533R9 = false

  int DummyInt;
  float DummyFloat;
  double DummyDouble;
  long double DummyLongDouble;

  ASSERT_CONSTEXPR_CXX23(std::abs(-1) == 1);
  ASSERT_NOT_CONSTEXPR_CXX23(std::abs(-1L) == 1L);
  ASSERT_NOT_CONSTEXPR_CXX23(std::abs(-1LL) == 1LL);
  ASSERT_NOT_CONSTEXPR_CXX23(std::abs(-1.0f) == 1.0f);
  ASSERT_NOT_CONSTEXPR_CXX23(std::abs(-1.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::abs(-1.0L) == 1.0L);

  ASSERT_CONSTEXPR_CXX23(std::labs(-1L) == 1L);
  ASSERT_CONSTEXPR_CXX23(std::llabs(-1LL) == 1LL);

  ASSERT_NOT_CONSTEXPR_CXX23(std::div(13, 5).rem == 3);
  ASSERT_NOT_CONSTEXPR_CXX23(std::div(13L, 5L).rem == 3L);
  ASSERT_NOT_CONSTEXPR_CXX23(std::div(13LL, 5LL).rem == 3LL);
  ASSERT_NOT_CONSTEXPR_CXX23(std::ldiv(13L, 5L).rem == 3L);
  ASSERT_NOT_CONSTEXPR_CXX23(std::lldiv(13LL, 5LL).rem == 3LL);

  ASSERT_NOT_CONSTEXPR_CXX23(std::frexp(0.0f, &DummyInt) == 0.0f);
  ASSERT_NOT_CONSTEXPR_CXX23(std::frexp(0.0, &DummyInt) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::frexp(0.0L, &DummyInt) == 0.0L);
  ASSERT_NOT_CONSTEXPR_CXX23(std::frexpf(0.0f, &DummyInt) == 0.0f);
  ASSERT_NOT_CONSTEXPR_CXX23(std::frexpl(0.0L, &DummyInt) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::ilogb(1.0f) == 0);
  ASSERT_CONSTEXPR_CXX23(std::ilogb(1.0) == 0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::ilogb(1.0L) == 0);
  ASSERT_CONSTEXPR_CXX23(std::ilogbf(1.0f) == 0);
  ASSERT_CONSTEXPR_CXX23(std::ilogbl(1.0L) == 0);

  ASSERT_NOT_CONSTEXPR_CXX23(std::ldexp(1.0f, 1) == 2.0f);
  ASSERT_CONSTEXPR_CXX23(std::ldexp(1.0, 1) == 2.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::ldexp(1.0L, 1) == 2.0L);
  ASSERT_CONSTEXPR_CXX23(std::ldexpf(1.0f, 1) == 2.0f);
  ASSERT_CONSTEXPR_CXX23(std::ldexpl(1.0L, 1) == 2.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::logb(1.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX23(std::logb(1.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::logb(1.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX23(std::logbf(1.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX23(std::logbl(1.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::modf(1.0f, &DummyFloat) == 0.0f);
  ASSERT_NOT_CONSTEXPR_CXX23(std::modf(1.0, &DummyDouble) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::modf(1.0L, &DummyLongDouble) == 0.0L);
  ASSERT_NOT_CONSTEXPR_CXX23(std::modff(1.0f, &DummyFloat) == 0.0f);
  ASSERT_NOT_CONSTEXPR_CXX23(std::modfl(1.0L, &DummyLongDouble) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::scalbn(1.0f, 1) == 2.0f);
  ASSERT_CONSTEXPR_CXX23(std::scalbn(1.0, 1) == 2.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::scalbn(1.0L, 1) == 2.0L);
  ASSERT_CONSTEXPR_CXX23(std::scalbnf(1.0f, 1) == 2.0f);
  ASSERT_CONSTEXPR_CXX23(std::scalbnl(1.0L, 1) == 2.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::scalbln(1.0f, 1L) == 2);
  ASSERT_CONSTEXPR_CXX23(std::scalbln(1.0, 1L) == 2.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::scalbln(1.0L, 1L) == 2.0L);
  ASSERT_CONSTEXPR_CXX23(std::scalblnf(1.0f, 1L) == 2.0f);
  ASSERT_CONSTEXPR_CXX23(std::scalblnl(1.0L, 1L) == 2.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::fabs(-1.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX23(std::fabs(-1.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::fabs(-1.0L) == 1.0L);
  ASSERT_CONSTEXPR_CXX23(std::fabsf(-1.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX23(std::fabsl(-1.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::ceil(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX23(std::ceil(0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::ceil(0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX23(std::ceilf(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX23(std::ceill(0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::floor(1.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX23(std::floor(1.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::floor(1.0L) == 1.0L);
  ASSERT_CONSTEXPR_CXX23(std::floorf(1.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX23(std::floorl(1.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::round(1.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX23(std::round(1.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::round(1.0L) == 1.0L);
  ASSERT_CONSTEXPR_CXX23(std::roundf(1.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX23(std::roundl(1.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::lround(1.0f) == 1L);
  ASSERT_CONSTEXPR_CXX23(std::lround(1.0) == 1L);
  ASSERT_NOT_CONSTEXPR_CXX23(std::lround(1.0L) == 1L);
  ASSERT_CONSTEXPR_CXX23(std::lroundf(1.0f) == 1L);
  ASSERT_CONSTEXPR_CXX23(std::lroundl(1.0L) == 1L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::llround(1.0f) == 1LL);
  ASSERT_CONSTEXPR_CXX23(std::llround(1.0) == 1LL);
  ASSERT_NOT_CONSTEXPR_CXX23(std::llround(1.0L) == 1LL);
  ASSERT_CONSTEXPR_CXX23(std::llroundf(1.0f) == 1LL);
  ASSERT_CONSTEXPR_CXX23(std::llroundl(1.0L) == 1LL);

  ASSERT_NOT_CONSTEXPR_CXX23(std::trunc(1.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX23(std::trunc(1.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::trunc(1.0L) == 1.0L);
  ASSERT_CONSTEXPR_CXX23(std::truncf(1.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX23(std::truncl(1.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::fmod(1.5f, 1.0f) == 0.5f);
  ASSERT_CONSTEXPR_CXX23(std::fmod(1.5, 1.0) == 0.5);
  ASSERT_NOT_CONSTEXPR_CXX23(std::fmod(1.5L, 1.0L) == 0.5L);
  ASSERT_CONSTEXPR_CXX23(std::fmodf(1.5f, 1.0f) == 0.5f);
  ASSERT_CONSTEXPR_CXX23(std::fmodl(1.5L, 1.0L) == 0.5L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::remainder(0.5f, 1.0f) == 0.5f);
  ASSERT_CONSTEXPR_CXX23(std::remainder(0.5, 1.0) == 0.5);
  ASSERT_NOT_CONSTEXPR_CXX23(std::remainder(0.5L, 1.0L) == 0.5L);
  ASSERT_CONSTEXPR_CXX23(std::remainderf(0.5f, 1.0f) == 0.5f);
  ASSERT_CONSTEXPR_CXX23(std::remainderl(0.5L, 1.0L) == 0.5L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::remquo(0.5f, 1.0f, &DummyInt) == 0.5f);
  ASSERT_NOT_CONSTEXPR_CXX23(std::remquo(0.5, 1.0, &DummyInt) == 0.5);
  ASSERT_NOT_CONSTEXPR_CXX23(std::remquo(0.5L, 1.0L, &DummyInt) == 0.5L);
  ASSERT_NOT_CONSTEXPR_CXX23(std::remquof(0.5f, 1.0f, &DummyInt) == 0.5f);
  ASSERT_NOT_CONSTEXPR_CXX23(std::remquol(0.5L, 1.0L, &DummyInt) == 0.5L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::copysign(1.0f, 1.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX23(std::copysign(1.0, 1.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::copysign(1.0L, 1.0L) == 1.0L);
  ASSERT_CONSTEXPR_CXX23(std::copysignf(1.0f, 1.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX23(std::copysignl(1.0L, 1.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::nextafter(0.0f, 0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX23(std::nextafter(0.0, 0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::nextafter(0.0L, 0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX23(std::nextafterf(0.0f, 0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX23(std::nextafterl(0.0L, 0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::nexttoward(0.0f, 0.0L) == 0.0f);
  ASSERT_CONSTEXPR_CXX23(std::nexttoward(0.0, 0.0L) == 0.0f);
  ASSERT_NOT_CONSTEXPR_CXX23(std::nexttoward(0.0L, 0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX23(std::nexttowardf(0.0f, 0.0L) == 0.0f);
  ASSERT_CONSTEXPR_CXX23(std::nexttowardl(0.0L, 0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::fdim(1.0f, 0.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX23(std::fdim(1.0, 0.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::fdim(1.0L, 0.0L) == 1.0L);
  ASSERT_CONSTEXPR_CXX23(std::fdimf(1.0f, 0.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX23(std::fdiml(1.0L, 0.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::fmax(1.0f, 0.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX23(std::fmax(1.0, 0.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::fmax(1.0L, 0.0L) == 1.0L);
  ASSERT_CONSTEXPR_CXX23(std::fmaxf(1.0f, 0.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX23(std::fmaxl(1.0L, 0.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::fmin(1.0f, 0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX23(std::fmin(1.0, 0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::fmin(1.0L, 0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX23(std::fminf(1.0f, 0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX23(std::fminl(1.0L, 0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::fma(1.0f, 1.0f, 1.0f) == 2.0f);
  ASSERT_CONSTEXPR_CXX23(std::fma(1.0, 1.0, 1.0) == 2.0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::fma(1.0L, 1.0L, 1.0L) == 2.0L);
  ASSERT_CONSTEXPR_CXX23(std::fmaf(1.0f, 1.0f, 1.0f) == 2.0f);
  ASSERT_CONSTEXPR_CXX23(std::fmal(1.0L, 1.0L, 1.0L) == 2.0L);

  ASSERT_NOT_CONSTEXPR_CXX23(std::fpclassify(-1.0f) == FP_NORMAL);
  ASSERT_NOT_CONSTEXPR_CXX23(std::fpclassify(-1.0) == FP_NORMAL);
  ASSERT_NOT_CONSTEXPR_CXX23(std::fpclassify(-1.0L) == FP_NORMAL);

  ASSERT_NOT_CONSTEXPR_CXX23(std::isfinite(-1.0f) == 1);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isfinite(-1.0) == 1);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isfinite(-1.0L) == 1);

  ASSERT_NOT_CONSTEXPR_CXX23(std::isinf(-1.0f) == 0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isinf(-1.0) == 0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isinf(-1.0L) == 0);

  ASSERT_NOT_CONSTEXPR_CXX23(std::isnan(-1.0f) == 0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isnan(-1.0) == 0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isnan(-1.0L) == 0);

  ASSERT_NOT_CONSTEXPR_CXX23(std::isnormal(-1.0f) == 1);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isnormal(-1.0) == 1);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isnormal(-1.0L) == 1);

  ASSERT_NOT_CONSTEXPR_CXX23(std::signbit(-1.0f) == 1);
  ASSERT_NOT_CONSTEXPR_CXX23(std::signbit(-1.0) == 1);
  ASSERT_NOT_CONSTEXPR_CXX23(std::signbit(-1.0L) == 1);

  ASSERT_NOT_CONSTEXPR_CXX23(std::isgreater(-1.0f, 0.0f) == 0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isgreater(-1.0, 0.0) == 0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isgreater(-1.0L, 0.0L) == 0);

  ASSERT_NOT_CONSTEXPR_CXX23(std::isgreaterequal(-1.0f, 0.0f) == 0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isgreaterequal(-1.0, 0.0) == 0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isgreaterequal(-1.0L, 0.0L) == 0);

  ASSERT_NOT_CONSTEXPR_CXX23(std::isless(-1.0f, 0.0f) == 1);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isless(-1.0, 0.0) == 1);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isless(-1.0L, 0.0L) == 1);

  ASSERT_NOT_CONSTEXPR_CXX23(std::islessequal(-1.0f, 0.0f) == 1);
  ASSERT_NOT_CONSTEXPR_CXX23(std::islessequal(-1.0, 0.0) == 1);
  ASSERT_NOT_CONSTEXPR_CXX23(std::islessequal(-1.0L, 0.0L) == 1);

  ASSERT_NOT_CONSTEXPR_CXX23(std::islessgreater(-1.0f, 0.0f) == 1);
  ASSERT_NOT_CONSTEXPR_CXX23(std::islessgreater(-1.0, 0.0) == 1);
  ASSERT_NOT_CONSTEXPR_CXX23(std::islessgreater(-1.0L, 0.0L) == 1);

  ASSERT_NOT_CONSTEXPR_CXX23(std::isunordered(-1.0f, 0.0f) == 0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isunordered(-1.0, 0.0) == 0);
  ASSERT_NOT_CONSTEXPR_CXX23(std::isunordered(-1.0L, 0.0L) == 0);

  assert(!ImplementedP0533R9 && R"(
Congratulations! You just have implemented P0533R9 (https://wg21.link/p0533r9).
Please go to `clang/www/cxx_status.html` and change the paper's implementation
status. Also please delete this assert and refactor `ASSERT_CONSTEXPR_CXX23`
and `ASSERT_NOT_CONSTEXPR_CXX23`.
)");

  return 0;
}
