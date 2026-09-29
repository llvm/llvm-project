//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// REQUIRES: clang

// We don't control the implementation of these functions on windows
// UNSUPPORTED: windows

// Missing some math functions.
// XFAIL: LLVM-LIBC-FIXME

// Check that Clang supports constexpr <cmath> and <cstdlib> functions
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

  // acos()
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

  // atan2()
  ASSERT_NOT_CONSTEXPR_CXX26(std::atan2(0.f, 1.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::atan2(0.0, 1.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::atan2(0.0L, 1.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::atan2f(0.f, 1.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::atan2l(0.0L, 1.0L) == 0.0L);

  // cos()
  ASSERT_NOT_CONSTEXPR_CXX26(std::cos(0.f) == 1.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::cos(0.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::cos(0.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::cosf(0.f) == 1.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::cosl(0.0L) == 1.0L);

  // sin()
  ASSERT_NOT_CONSTEXPR_CXX26(std::sin(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::sin(0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::sin(0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::sinf(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::sinl(0.0L) == 0.0L);

  // tan()
  ASSERT_NOT_CONSTEXPR_CXX26(std::tan(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::tan(0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::tan(0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::tanf(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::tanl(0.0L) == 0.0L);

  // acosh()
  ASSERT_NOT_CONSTEXPR_CXX26(std::acosh(1.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::acosh(1.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::acosh(1.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::acoshf(1.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::acoshl(1.0L) == 0.0L);

  // asinh()
  ASSERT_NOT_CONSTEXPR_CXX26(std::asinh(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::asinh(0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::asinh(0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::asinhf(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::asinhl(0.0L) == 0.0L);

  // atanh()
  ASSERT_NOT_CONSTEXPR_CXX26(std::atanh(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::atanh(0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::atanh(0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::atanhf(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::atanhl(0.0L) == 0.0L);

  // cosh()
  ASSERT_NOT_CONSTEXPR_CXX26(std::cosh(0.f) == 1.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::cosh(0.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::cosh(0.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::coshf(0.f) == 1.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::coshl(0.0L) == 1.0L);

  // sinh()
  ASSERT_NOT_CONSTEXPR_CXX26(std::sinh(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::sinh(0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::sinh(0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::sinhf(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::sinhl(0.0L) == 0.0L);

  // tanh()
  ASSERT_NOT_CONSTEXPR_CXX26(std::tanh(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::tanh(0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::tanh(0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::tanhf(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::tanhl(0.0L) == 0.0L);

  // exp()
  ASSERT_NOT_CONSTEXPR_CXX26(std::exp(0.f) == 1.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::exp(0.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::exp(0.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::expf(0.f) == 1.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::expl(0.0L) == 1.0L);

  // exp2()
  ASSERT_NOT_CONSTEXPR_CXX26(std::exp2(0.f) == 1.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::exp2(0.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::exp2(0.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::exp2f(0.f) == 1.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::exp2l(0.0L) == 1.0L);

  // expm1()
  ASSERT_NOT_CONSTEXPR_CXX26(std::expm1(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::expm1(0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::expm1(0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::expm1f(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::expm1l(0.0L) == 0.0L);

  // log()
  ASSERT_NOT_CONSTEXPR_CXX26(std::log(1.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::log(1.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::log(1.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::logf(1.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::logl(1.0L) == 0.0L);

  // log10()
  ASSERT_NOT_CONSTEXPR_CXX26(std::log10(1.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::log10(1.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::log10(1.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::log10f(1.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::log10l(1.0L) == 0.0L);

  // log1p()
  ASSERT_NOT_CONSTEXPR_CXX26(std::log1p(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::log1p(0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::log1p(0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::log1pf(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::log1pl(0.0L) == 0.0L);

  // log2()
  ASSERT_NOT_CONSTEXPR_CXX26(std::log2(1.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::log2(1.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::log2(1.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::log2f(1.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::log2l(1.0L) == 0.0L);

  // cbrt()
  ASSERT_NOT_CONSTEXPR_CXX26(std::cbrt(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::cbrt(0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::cbrt(0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::cbrtf(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::cbrtl(0.0L) == 0.0L);

  // [c.math.abs], absolute values

  // hypot()
  ASSERT_NOT_CONSTEXPR_CXX26(std::hypot(0.f, 0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::hypot(0.0, 0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::hypot(0.0L, 0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::hypotf(0.f, 0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::hypotl(0.0L, 0.0L) == 0.0L);

  // [c.math.hypot3], three-dimensional hypotenuse

  // hypot() - three arguments
  ASSERT_NOT_CONSTEXPR_CXX26(std::hypot(0.f, 0.f, 0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::hypot(0.0, 0.0, 0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::hypot(0.0L, 0.0L, 0.0L) == 0.0L);

  // pow()
  ASSERT_NOT_CONSTEXPR_CXX26(std::pow(0.f, 0.f) == 1.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::pow(0.0, 0.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::pow(0.0L, 0.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::powf(0.f, 0.f) == 1.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::powl(0.0L, 0.0L) == 1.0L);

  // sqrt()
  ASSERT_NOT_CONSTEXPR_CXX26(std::sqrt(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::sqrt(0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::sqrt(0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::sqrtf(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::sqrtl(0.0L) == 0.0L);

  // erf()
  ASSERT_NOT_CONSTEXPR_CXX26(std::erf(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::erf(0.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::erf(0.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::erff(0.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::erfl(0.0L) == 0.0L);

  // erfc()
  ASSERT_NOT_CONSTEXPR_CXX26(std::erfc(0.f) == 1.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::erfc(0.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::erfc(0.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::erfcf(0.f) == 1.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::erfcl(0.0L) == 1.0L);

  // lgamma()
  ASSERT_NOT_CONSTEXPR_CXX26(std::lgamma(1.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::lgamma(1.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::lgamma(1.0L) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::lgammaf(1.f) == 0.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::lgammal(1.0L) == 0.0L);

  // tgamma()
  ASSERT_NOT_CONSTEXPR_CXX26(std::tgamma(1.f) == 1.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::tgamma(1.0) == 1.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::tgamma(1.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::tgammaf(1.f) == 1.f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::tgammal(1.0L) == 1.0L);

  assert(!ImplementedP1383R2 && R"(
Congratulations! You just have implemented P1383R2 (https://wg21.link/P1383R2).
Please go to `clang/www/cxx_status.html` and change the paper's implementation
status. Also please delete this assert and refactor `ASSERT_CONSTEXPR_CXX26`
and `ASSERT_NOT_CONSTEXPR_CXX26`.
)");

  return 0;
}
