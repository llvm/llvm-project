//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that GCC supports constexpr <cmath> and <complex> functions
// mentioned in the P1383R2 paper that is part of C++26
// (https://wg21.link/p1383r2)
//
// Every function called in this test should become constexpr. Whenever some
// of the desired function become constexpr, the programmer switches
// `ASSERT_NOT_CONSTEXPR_CXX26` to `ASSERT_CONSTEXPR_CXX26` and eventually the
// paper is implemented in libc++.
// The test also works as a reference list of unimplemented functions.
//
// REQUIRES: gcc
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

#include <cassert>
#include <cmath>
#include <complex>

int main(int, char**) {
  bool ImplementedP1383R2 = true;

#define ASSERT_CONSTEXPR_CXX26(Expr) static_assert(__builtin_constant_p(Expr) && (Expr))
#define ASSERT_NOT_CONSTEXPR_CXX26(Expr)                                                                               \
  static_assert(!__builtin_constant_p(Expr));                                                                          \
  assert(Expr);                                                                                                        \
  ImplementedP1383R2 = false

  ASSERT_CONSTEXPR_CXX26(std::acos(1.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::acos(1.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::acos(1.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::acosf(1.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::acosl(1.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::asin(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::asin(0.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::asin(0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::asinf(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::asinl(0.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::atan(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::atan(0.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::atan(0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::atanf(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::atanl(0.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::atan2(0.0f, 1.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::atan2(0.0, 1.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::atan2(0.0L, 1.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::atan2f(0.0f, 1.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::atan2l(0.0L, 1.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::cos(0.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX26(std::cos(0.0) == 1.0);
  ASSERT_CONSTEXPR_CXX26(std::cos(0.0L) == 1.0L);
  ASSERT_CONSTEXPR_CXX26(std::cosf(0.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX26(std::cosl(0.0L) == 1.0L);

  ASSERT_CONSTEXPR_CXX26(std::sin(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::sin(0.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::sin(0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::sinf(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::sinl(0.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::tan(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::tan(0.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::tan(0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::tanf(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::tanl(0.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::acosh(1.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::acosh(1.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::acosh(1.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::acoshf(1.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::acoshl(1.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::asinh(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::asinh(0.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::asinh(0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::asinhf(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::asinhl(0.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::atanh(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::atanh(0.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::atanh(0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::atanhf(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::atanhl(0.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::cosh(0.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX26(std::cosh(0.0) == 1.0);
  ASSERT_CONSTEXPR_CXX26(std::cosh(0.0L) == 1.0L);
  ASSERT_CONSTEXPR_CXX26(std::coshf(0.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX26(std::coshl(0.0L) == 1.0L);

  ASSERT_CONSTEXPR_CXX26(std::sinh(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::sinh(0.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::sinh(0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::sinhf(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::sinhl(0.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::tanh(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::tanh(0.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::tanh(0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::tanhf(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::tanhl(0.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::exp(0.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX26(std::exp(0.0) == 1.0);
  ASSERT_CONSTEXPR_CXX26(std::exp(0.0L) == 1.0L);
  ASSERT_CONSTEXPR_CXX26(std::expf(0.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX26(std::expl(0.0L) == 1.0L);

  ASSERT_CONSTEXPR_CXX26(std::exp2(3.0f) == 8.0f);
  ASSERT_CONSTEXPR_CXX26(std::exp2(3.0) == 8.0);
  ASSERT_CONSTEXPR_CXX26(std::exp2(3.0L) == 8.0L);
  ASSERT_CONSTEXPR_CXX26(std::exp2f(3.0f) == 8.0f);
  ASSERT_CONSTEXPR_CXX26(std::exp2l(3.0L) == 8.0L);

  ASSERT_CONSTEXPR_CXX26(std::expm1(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::expm1(0.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::expm1(0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::expm1f(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::expm1l(0.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::log(1.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::log(1.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::log(1.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::logf(1.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::logl(1.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::log10(1.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::log10(1.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::log10(1.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::log10f(1.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::log10l(1.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::log1p(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::log1p(0.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::log1p(0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::log1pf(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::log1pl(0.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::log2(1.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::log2(1.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::log2(1.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::log2f(1.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::log2l(1.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::cbrt(8.0f) == 2.0f);
  ASSERT_CONSTEXPR_CXX26(std::cbrt(8.0) == 2.0);
  ASSERT_CONSTEXPR_CXX26(std::cbrt(8.0L) == 2.0L);
  ASSERT_CONSTEXPR_CXX26(std::cbrtf(8.0f) == 2.0f);
  ASSERT_CONSTEXPR_CXX26(std::cbrtl(8.0L) == 2.0L);

  ASSERT_CONSTEXPR_CXX26(std::hypot(3.0f, 4.0f) == 5.0f);
  ASSERT_CONSTEXPR_CXX26(std::hypot(3.0, 4.0) == 5.0);
  ASSERT_CONSTEXPR_CXX26(std::hypot(3.0L, 4.0L) == 5.0L);
  ASSERT_CONSTEXPR_CXX26(std::hypotf(3.0f, 4.0f) == 5.0f);
  ASSERT_CONSTEXPR_CXX26(std::hypotl(3.0L, 4.0L) == 5.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::hypot(0.0f, 3.0f, 4.0f) == 5.0f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::hypot(0.0, 3.0, 4.0) == 5.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::hypot(0.0L, 3.0L, 4.0L) == 5.0L);

  ASSERT_CONSTEXPR_CXX26(std::pow(2.0f, 3.0f) == 8.0f);
  ASSERT_CONSTEXPR_CXX26(std::pow(2.0, 3.0) == 8.0);
  ASSERT_CONSTEXPR_CXX26(std::pow(2.0L, 3.0L) == 8.0L);
  ASSERT_CONSTEXPR_CXX26(std::powf(2.0f, 3.0f) == 8.0f);
  ASSERT_CONSTEXPR_CXX26(std::powl(2.0L, 3.0L) == 8.0L);

  ASSERT_CONSTEXPR_CXX26(std::sqrt(4.0f) == 2.0f);
  ASSERT_CONSTEXPR_CXX26(std::sqrt(4.0) == 2.0);
  ASSERT_CONSTEXPR_CXX26(std::sqrt(4.0L) == 2.0L);
  ASSERT_CONSTEXPR_CXX26(std::sqrtf(4.0f) == 2.0f);
  ASSERT_CONSTEXPR_CXX26(std::sqrtl(4.0L) == 2.0L);

  ASSERT_CONSTEXPR_CXX26(std::erf(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::erf(0.0) == 0.0);
  ASSERT_CONSTEXPR_CXX26(std::erf(0.0L) == 0.0L);
  ASSERT_CONSTEXPR_CXX26(std::erff(0.0f) == 0.0f);
  ASSERT_CONSTEXPR_CXX26(std::erfl(0.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::erfc(0.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX26(std::erfc(0.0) == 1.0);
  ASSERT_CONSTEXPR_CXX26(std::erfc(0.0L) == 1.0L);
  ASSERT_CONSTEXPR_CXX26(std::erfcf(0.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX26(std::erfcl(0.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::lgamma(1.0f) == 0.0f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::lgamma(1.0) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::lgamma(1.0L) == 0.0L);
  ASSERT_NOT_CONSTEXPR_CXX26(std::lgammaf(1.0f) == 0.0f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::lgammal(1.0L) == 0.0L);

  ASSERT_CONSTEXPR_CXX26(std::tgamma(1.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX26(std::tgamma(1.0) == 1.0);
  ASSERT_CONSTEXPR_CXX26(std::tgamma(1.0L) == 1.0L);
  ASSERT_CONSTEXPR_CXX26(std::tgammaf(1.0f) == 1.0f);
  ASSERT_CONSTEXPR_CXX26(std::tgammal(1.0L) == 1.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::abs(std::complex<float>(3, 4)) == 5.0f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::abs(std::complex<double>(3, 4)) == 5.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::abs(std::complex<long double>(3, 4)) == 5.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::arg(std::complex<float>(1, 0)) == 0.0f);
  ASSERT_NOT_CONSTEXPR_CXX26(std::arg(std::complex<double>(1, 0)) == 0.0);
  ASSERT_NOT_CONSTEXPR_CXX26(std::arg(std::complex<long double>(1, 0)) == 0.0L);

  ASSERT_NOT_CONSTEXPR_CXX26(std::proj(std::complex<float>(1, 2)) == std::complex<float>(1, 2));
  ASSERT_NOT_CONSTEXPR_CXX26(std::proj(std::complex<double>(1, 2)) == std::complex<double>(1, 2));
  ASSERT_NOT_CONSTEXPR_CXX26(std::proj(std::complex<long double>(1, 2)) == std::complex<long double>(1, 2));

  ASSERT_NOT_CONSTEXPR_CXX26(std::polar(1.0f, 0.0f) == std::complex<float>(1, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::polar(1.0, 0.0) == std::complex<double>(1, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::polar(1.0L, 0.0L) == std::complex<long double>(1, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::acos(std::complex<float>(1, 0)) == std::complex<float>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::acos(std::complex<double>(1, 0)) == std::complex<double>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::acos(std::complex<long double>(1, 0)) == std::complex<long double>(0, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::asin(std::complex<float>(0, 0)) == std::complex<float>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::asin(std::complex<double>(0, 0)) == std::complex<double>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::asin(std::complex<long double>(0, 0)) == std::complex<long double>(0, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::atan(std::complex<float>(0, 0)) == std::complex<float>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::atan(std::complex<double>(0, 0)) == std::complex<double>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::atan(std::complex<long double>(0, 0)) == std::complex<long double>(0, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::acosh(std::complex<float>(1, 0)) == std::complex<float>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::acosh(std::complex<double>(1, 0)) == std::complex<double>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::acosh(std::complex<long double>(1, 0)) == std::complex<long double>(0, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::asinh(std::complex<float>(0, 0)) == std::complex<float>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::asinh(std::complex<double>(0, 0)) == std::complex<double>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::asinh(std::complex<long double>(0, 0)) == std::complex<long double>(0, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::atanh(std::complex<float>(0, 0)) == std::complex<float>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::atanh(std::complex<double>(0, 0)) == std::complex<double>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::atanh(std::complex<long double>(0, 0)) == std::complex<long double>(0, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::cos(std::complex<float>(0, 0)) == std::complex<float>(1, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::cos(std::complex<double>(0, 0)) == std::complex<double>(1, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::cos(std::complex<long double>(0, 0)) == std::complex<long double>(1, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::cosh(std::complex<float>(0, 0)) == std::complex<float>(1, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::cosh(std::complex<double>(0, 0)) == std::complex<double>(1, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::cosh(std::complex<long double>(0, 0)) == std::complex<long double>(1, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::exp(std::complex<float>(0, 0)) == std::complex<float>(1, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::exp(std::complex<double>(0, 0)) == std::complex<double>(1, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::exp(std::complex<long double>(0, 0)) == std::complex<long double>(1, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::log(std::complex<float>(1, 0)) == std::complex<float>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::log(std::complex<double>(1, 0)) == std::complex<double>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::log(std::complex<long double>(1, 0)) == std::complex<long double>(0, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::log10(std::complex<float>(1, 0)) == std::complex<float>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::log10(std::complex<double>(1, 0)) == std::complex<double>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::log10(std::complex<long double>(1, 0)) == std::complex<long double>(0, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(
      std::pow(std::complex<float>(1, 0), std::complex<float>(3, 0)) == std::complex<float>(1, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(
      std::pow(std::complex<double>(1, 0), std::complex<double>(3, 0)) == std::complex<double>(1, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(
      std::pow(std::complex<long double>(1, 0), std::complex<long double>(3, 0)) == std::complex<long double>(1, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::pow(std::complex<float>(1, 0), 3.0f) == std::complex<float>(1, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::pow(std::complex<double>(1, 0), 3.0) == std::complex<double>(1, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::pow(std::complex<long double>(1, 0), 3.0L) == std::complex<long double>(1, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::pow(1.0f, std::complex<float>(3, 0)) == std::complex<float>(1, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::pow(1.0, std::complex<double>(3, 0)) == std::complex<double>(1, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::pow(1.0L, std::complex<long double>(3, 0)) == std::complex<long double>(1, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::sin(std::complex<float>(0, 0)) == std::complex<float>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::sin(std::complex<double>(0, 0)) == std::complex<double>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::sin(std::complex<long double>(0, 0)) == std::complex<long double>(0, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::sinh(std::complex<float>(0, 0)) == std::complex<float>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::sinh(std::complex<double>(0, 0)) == std::complex<double>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::sinh(std::complex<long double>(0, 0)) == std::complex<long double>(0, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::sqrt(std::complex<float>(4, 0)) == std::complex<float>(2, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::sqrt(std::complex<double>(4, 0)) == std::complex<double>(2, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::sqrt(std::complex<long double>(4, 0)) == std::complex<long double>(2, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::tan(std::complex<float>(0, 0)) == std::complex<float>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::tan(std::complex<double>(0, 0)) == std::complex<double>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::tan(std::complex<long double>(0, 0)) == std::complex<long double>(0, 0));

  ASSERT_NOT_CONSTEXPR_CXX26(std::tanh(std::complex<float>(0, 0)) == std::complex<float>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::tanh(std::complex<double>(0, 0)) == std::complex<double>(0, 0));
  ASSERT_NOT_CONSTEXPR_CXX26(std::tanh(std::complex<long double>(0, 0)) == std::complex<long double>(0, 0));

  assert(!ImplementedP1383R2 && R"(
Congratulations! You just have implemented P1383R2 (https://wg21.link/p1383r2).
Please go to `clang/www/cxx_status.html` and change the paper's implementation
status. Also please delete this assert and refactor `ASSERT_CONSTEXPR_CXX26`
and `ASSERT_NOT_CONSTEXPR_CXX26`.
)");

  return 0;
}
