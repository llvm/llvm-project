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

// check that const cmath functions are declared [[nodiscard]]

#include <cmath>

void test() {
  std::signbit(0.f);             // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::signbit(0.);              // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::signbit(0.l);             // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::signbit(0);               // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::signbit(0U);              // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::fpclassify(0.f);          // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fpclassify(0.);           // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fpclassify(0.l);          // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fpclassify(0);            // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fpclassify(0U);           // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::isfinite(0.f);            // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isfinite(0.);             // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isfinite(0.l);            // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isfinite(0);              // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isfinite(0U);             // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::isinf(0.f);               // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isinf(0.);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isinf(0.l);               // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isinf(0);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isinf(0U);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::isnan(0.f);               // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isnan(0.);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isnan(0.l);               // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isnan(0);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isnan(0U);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::isnormal(0.f);            // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isnormal(0.);             // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isnormal(0.l);            // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isnormal(0);              // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isnormal(0U);             // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::isgreater(0.f, 0.f);      // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isgreater(0., 0.);        // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isgreater(0.l, 0.l);      // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isgreater(0, 0);          // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isgreater(0U, 0U);        // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::isgreaterequal(0.f, 0.f); // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isgreaterequal(0., 0.);   // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isgreaterequal(0.l, 0.l); // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isgreaterequal(0, 0);     // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isgreaterequal(0U, 0U);   // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::isless(0.f, 0.f);         // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isless(0., 0.);           // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isless(0.l, 0.l);         // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isless(0, 0);             // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isless(0U, 0U);           // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::islessequal(0.f, 0.f);    // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::islessequal(0., 0.);      // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::islessequal(0.l, 0.l);    // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::islessequal(0, 0);        // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::islessequal(0U, 0U);      // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::islessgreater(0.f, 0.f);  // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::islessgreater(0., 0.);    // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::islessgreater(0.l, 0.l);  // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::islessgreater(0, 0);      // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::islessgreater(0U, 0U);    // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::isunordered(0.f, 0.f);    // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isunordered(0., 0.);      // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isunordered(0.l, 0.l);    // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isunordered(0, 0);        // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::isunordered(0U, 0U);      // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::ceil(0.f);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::ceil(0.);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::ceil(0.l);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::ceil(0);                  // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::ceil(0U);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::fabs(0.f);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fabs(0.);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fabs(0.l);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fabs(0);                  // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fabs(0U);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::floor(0.f);               // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::floor(0.);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::floor(0.l);               // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::floor(0);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::floor(0U);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::cbrt(0.f);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::cbrt(0.);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::cbrt(0.l);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::cbrt(0);                  // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::cbrt(0U);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::copysign(0.f, 0.f);       // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::copysign(0., 0.);         // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::copysign(0.l, 0.l);       // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::copysign(0, 0);           // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::copysign(0U, 0U);         // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::fmax(0.f, 0.f);           // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fmax(0., 0.);             // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fmax(0.l, 0.l);           // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fmax(0, 0);               // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fmax(0U, 0U);             // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::fmin(0.f, 0.f);           // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fmin(0., 0.);             // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fmin(0.l, 0.l);           // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fmin(0, 0);               // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::fmin(0U, 0U);             // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::nearbyint(0.f);           // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::nearbyint(0.);            // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::nearbyint(0.l);           // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::nearbyint(0);             // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::nearbyint(0U);            // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::rint(0.f);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::rint(0.);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::rint(0.l);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::rint(0);                  // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::rint(0U);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::round(0.f);               // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::round(0.);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::round(0.l);               // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::round(0);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::round(0U);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}

  std::trunc(0.f);               // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::trunc(0.);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::trunc(0.l);               // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::trunc(0);                 // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
  std::trunc(0U);                // expected-warning-re {{ignoring return value of function declared with {{.*}} attribute}}
}
