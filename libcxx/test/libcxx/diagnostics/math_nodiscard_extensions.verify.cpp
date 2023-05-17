//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// We don't control the implementation of the math.h functions on windows
// UNSUPPORTED: windows

// Check that functions from `<cmath>` that Clang marks with the `[[gnu::const]]` attribute are declared
// `[[nodiscard]]`.

#include <cmath>
#include "test_macros.h"

void test() {
  // These tests rely on Clang's behaviour of adding `[[gnu::const]]` to the double overload of most of the functions
  // below.
  // Without that attribute being added implicitly, this test can't be checked consistently because its result depends
  // on whether we're getting libc++'s own `std::foo(double)` or the underlying C library's `foo(double)`.
#ifdef TEST_COMPILER_CLANG
  std::ceil(0.);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fabs(0.);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::floor(0.);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::cbrt(0.);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::copysign(0., 0.);         // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fmax(0., 0.);             // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fmin(0., 0.);             // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::nearbyint(0.);            // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::rint(0.);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::round(0.);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::trunc(0.);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
#endif

  std::signbit(0.f);             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::signbit(0.);              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::signbit(0.l);             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::signbit(0);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::signbit(0U);              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::fpclassify(0.f);          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fpclassify(0.);           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fpclassify(0.l);          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fpclassify(0);            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fpclassify(0U);           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::isfinite(0.f);            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isfinite(0.);             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isfinite(0.l);            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isfinite(0);              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isfinite(0U);             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::isinf(0.f);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isinf(0.);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isinf(0.l);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isinf(0);                 // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isinf(0U);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::isnan(0.f);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isnan(0.);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isnan(0.l);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isnan(0);                 // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isnan(0U);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::isnormal(0.f);            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isnormal(0.);             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isnormal(0.l);            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isnormal(0);              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isnormal(0U);             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::isgreater(0.f, 0.f);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isgreater(0., 0.);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isgreater(0.l, 0.l);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isgreater(0, 0);          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isgreater(0U, 0U);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::isgreaterequal(0.f, 0.f); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isgreaterequal(0., 0.);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isgreaterequal(0.l, 0.l); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isgreaterequal(0, 0);     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isgreaterequal(0U, 0U);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::isless(0.f, 0.f);         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isless(0., 0.);           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isless(0.l, 0.l);         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isless(0, 0);             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isless(0U, 0U);           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::islessequal(0.f, 0.f);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::islessequal(0., 0.);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::islessequal(0.l, 0.l);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::islessequal(0, 0);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::islessequal(0U, 0U);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::islessgreater(0.f, 0.f);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::islessgreater(0., 0.);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::islessgreater(0.l, 0.l);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::islessgreater(0, 0);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::islessgreater(0U, 0U);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::isunordered(0.f, 0.f);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isunordered(0., 0.);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isunordered(0.l, 0.l);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isunordered(0, 0);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::isunordered(0U, 0U);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::ceil(0.f);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ceil(0.l);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ceil(0);                  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ceil(0U);                 // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::fabs(0.f);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fabs(0.l);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fabs(0);                  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fabs(0U);                 // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::floor(0.f);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::floor(0.l);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::floor(0);                 // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::floor(0U);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::cbrt(0.f);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cbrt(0.l);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cbrt(0);                  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::cbrt(0U);                 // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::copysign(0.f, 0.f);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::copysign(0.l, 0.l);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::copysign(0, 0);           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::copysign(0U, 0U);         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::fmax(0.f, 0.f);           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fmax(0.l, 0.l);           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fmax(0, 0);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fmax(0U, 0U);             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::fmin(0.f, 0.f);           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fmin(0.l, 0.l);           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fmin(0, 0);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::fmin(0U, 0U);             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::nearbyint(0.f);           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::nearbyint(0.l);           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::nearbyint(0);             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::nearbyint(0U);            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::rint(0.f);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::rint(0.l);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::rint(0);                  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::rint(0U);                 // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::round(0.f);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::round(0.l);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::round(0);                 // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::round(0U);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::trunc(0.f);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::trunc(0.l);               // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::trunc(0);                 // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::trunc(0U);                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
