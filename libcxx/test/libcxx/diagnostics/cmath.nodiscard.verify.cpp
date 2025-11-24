//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// We don't control the implementation of the math.h functions on windows
// UNSUPPORTED: windows

// Check that functions are marked `[[nodiscard]]`.
// Check that functions from `<cmath>` that Clang marks with the `[[gnu::const]]` attribute are declared
// `[[nodiscard]]`.

#include <cmath>

#include "test_macros.h"

void test() {
  // Some tests rely on Clang's behaviour of adding `[[gnu::const]]` to the double overload of most of the functions
  // below. Without that attribute being added implicitly, this test can't be checked consistently because its result
  // depends on whether we're getting libc++'s own `std::foo(double)` or the underlying C library's `foo(double)`, e.g.
  // std::fabs(double).

  // Functions

  std::fabs(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fabs(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::fabs(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fabs(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Floating point manipulation functions

  std::copysign(0.F, 0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::copysign(0.L, 0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::copysign(0U, 0U);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Error functions

  std::erf(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::erf(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::erf(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::erf(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::erfc(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::erfc(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::erfc(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::erfc(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Exponential functions

  std::exp(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::exp(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::exp(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::exp(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Floating point manipulation functions

  int* iPtr = nullptr;

  std::frexp(0.F, iPtr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::frexp(0., iPtr); /// expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::frexp(0.L, iPtr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::frexp(0U, iPtr);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::ldexp(0.F, 2); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ldexp(0., 2);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::ldexp(0.L, 2); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ldexp(0U, 2);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Exponential functions

  std::exp2(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::exp2(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::exp2(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::exp2(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::expm1(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::expm1(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::expm1(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::expm1(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Floating point manipulation functions

  std::scalbln(0.F, 0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::scalbln(0., 0.L); // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::scalbln(0.L, 0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::scalbln(0U, 0.L);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::scalbn(0.F, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::scalbn(0., 0);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::scalbn(0.L, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::scalbn(0U, 0);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Power functions

  std::pow(0.F, 0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::pow(0., 0.);   // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::pow(0.L, 0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::pow(0U, 0U);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Basic operations

  std::fdim(0.F, 0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fdim(0., 0.);   // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::fdim(0.L, 0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fdim(0U, 0U);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::fma(0.F, 0.F, 0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fma(0., 0., 0.); // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::fma(0.L, 0.L, 0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fma(0U, 0U, 0U);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Error functions

  std::lgamma(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lgamma(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::lgamma(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lgamma(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::tgamma(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::tgamma(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::tgamma(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::tgamma(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Hyperbolic functions

  std::cosh(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cosh(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::cosh(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cosh(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::sinh(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::sinh(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::sinh(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::sinh(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::tanh(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::tanh(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::tanh(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::tanh(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Power functions

  std::hypot(0.F, 0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hypot(0., 0.);   // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::hypot(0.L, 0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hypot(0U, 0U);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 17
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hypot(0.F, 0.F, 0.F);
  // // expected-warning-re@+1 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::hypot(0., 0., 0.);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hypot(0.L, 0.L, 0.L);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hypot(0U, 0U, 0U);
#endif // _LIBCPP_STD_VER >= 17

  // Hyperbolic functions

  std::acosh(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::acosh(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::acosh(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::acosh(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::asinh(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::asinh(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::asinh(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::asinh(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::atanh(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::atanh(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::atanh(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::atanh(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Trigonometric functions

  std::acos(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::acos(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::acos(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::acos(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::asin(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::asin(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::asin(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::asin(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::atan(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::atan(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::atan(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::atan(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::atan2(0.F, 0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::atan2(0., 0.);   // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::atan2(0.L, 0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::atan2(0U, 0U);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Exponential functions

  std::log(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::log(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::log(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::log(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::log10(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::log10(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::log10(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::log10(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::ilogb(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ilogb(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::ilogb(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ilogb(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::log1p(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::log1p(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::log1p(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::log1p(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::log2(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::log2(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::log2(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::log2(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::logb(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::logb(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::logb(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::logb(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Basic operations

  std::fmax(0.F, 0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fmax(0., 0.);   // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::fmax(0.L, 0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fmax(0U, 0U);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::fmin(0.F, 0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fmin(0., 0.);   // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::fmin(0.L, 0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fmin(0U, 0U);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::fmod(0.F, 0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fmod(0., 0.);   // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::fmod(0.L, 0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fmod(0U, 0U);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  float* fPtr       = nullptr;
  double* dPtr      = nullptr;
  long double* lPtr = nullptr;

  std::modf(0.F, fPtr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::modf(0., dPtr);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::modf(0.L, lPtr); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::remainder(0.F, 0.F);
  // expected-warning-re@+1 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::remainder(0., 0.);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::remainder(0.L, 0.L);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::remainder(0U, 0U);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::remquo(0.F, 0.F, iPtr);
  // expected-warning-re@+1 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::remquo(0., 0., iPtr);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::remquo(0.L, 0.L, iPtr);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::remquo(0U, 0U, iPtr);

  // Power functions

  std::sqrt(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::sqrt(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::sqrt(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::sqrt(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::cbrt(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cbrt(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::cbrt(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cbrt(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Nearest integer floating point operations

  std::ceil(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ceil(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::ceil(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ceil(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::floor(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::floor(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::floor(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::floor(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::llrint(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::llrint(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::llrint(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::llrint(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::llround(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::llround(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::llround(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::llround(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::lrint(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lrint(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::lrint(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lrint(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::lround(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lround(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::lround(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lround(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::nearbyint(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::nearbyint(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::nearbyint(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::nearbyint(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::nextafter(0.F, 0.F);
  // expected-warning-re@+1 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::nextafter(0., 0.);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::nextafter(0.L, 0.L);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::nextafter(0U, 0U);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::nexttoward(0.F, 0.F);
  // expected-warning-re@+1 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::nexttoward(0., 0.);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::nexttoward(0.L, 0.L);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::nexttoward(0U, 0U);

  std::rint(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::rint(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::rint(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::rint(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::round(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::round(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::round(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::round(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::trunc(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::trunc(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::trunc(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::trunc(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 17
  // Mathematical special functions

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hermite(0U, 0.);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hermite(0U, 0.F);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hermite(0U, 0.L);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hermitef(0U, 0.F);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hermitel(0U, 0.L);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hermite(0U, 0U);
#endif // TEST_STD_VER >= 17

  std::signbit(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::signbit(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::signbit(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::signbit(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::isfinite(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isfinite(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::isfinite(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isfinite(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::isinf(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isinf(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::isinf(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isinf(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::isnan(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isnan(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::isnan(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isnan(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::isnormal(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isnormal(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::isnormal(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isnormal(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isgreater(0.F, 0.F);
  // expected-warning-re@+1 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::isgreater(0., 0.);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isgreater(0.L, 0.L);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isgreater(0U, 0U);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isgreaterequal(0.F, 0.F);
  // expected-warning-re@+1 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::isgreaterequal(0., 0.);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isgreaterequal(0.L, 0.L);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isgreaterequal(0U, 0U);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isless(0.F, 0.F);
  // expected-warning-re@+1 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::isless(0., 0.);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isless(0.L, 0.L);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isless(0U, 0U);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::islessequal(0.F, 0.F);
  // expected-warning-re@+1 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::islessequal(0., 0.);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::islessequal(0.L, 0.L);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::islessequal(0U, 0U);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::islessgreater(0.F, 0.F);
  // expected-warning-re@+1 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::islessgreater(0., 0.);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::islessgreater(0.L, 0.L);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::islessgreater(0U, 0U);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isunordered(0.F, 0.F);
  // expected-warning-re@+1 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::isunordered(0., 0.);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isunordered(0.L, 0.L);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isunordered(0U, 0U);

  // Trigonometric functions

  std::cos(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cos(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::cos(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cos(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::sin(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::sin(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::sin(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::sin(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::tan(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::tan(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::tan(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::tan(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // [c.math.lerp], linear interpolation

#if TEST_STD_VER >= 20
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lerp(0.F, 0.F, 0.F);
  // expected-warning-re@+1 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::lerp(0., 0., 0.);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lerp(0.L, 0.L, 0.L);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lerp(0U, 0U, 0U);
#endif

  std::fpclassify(0.F); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fpclassify(0.);  // expected-warning-re 0-1 {{ignoring return value of function declared with {{.*}} attribute}}
  std::fpclassify(0.L); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fpclassify(0U);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
