//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ADDITIONAL_COMPILE_FLAGS: -Wno-unused-value

#include <limits>

#include "type_algorithms.h"

void func() {
  std::numeric_limits<bool>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<bool>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<bool>::denorm_min();

  std::numeric_limits<int>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<int>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<int>::denorm_min();

  std::numeric_limits<float>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<float>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<float>::denorm_min();

  std::numeric_limits<double>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<double>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<double>::denorm_min();

  std::numeric_limits<long double>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<long double>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<long double>::denorm_min();

  std::numeric_limits<const bool>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<const bool>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<const bool>::denorm_min();

  std::numeric_limits<const int>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<const int>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<const int>::denorm_min();

  std::numeric_limits<const float>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<const float>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<const float>::denorm_min();

  std::numeric_limits<const double>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<const double>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<const double>::denorm_min();

  std::numeric_limits<const long double>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<const long double>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<const long double>::denorm_min();

  std::numeric_limits<volatile bool>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<volatile bool>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<volatile bool>::denorm_min();

  std::numeric_limits<volatile int>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<volatile int>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<volatile int>::denorm_min();

  std::numeric_limits<volatile float>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<volatile float>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<volatile float>::denorm_min();

  std::numeric_limits<volatile double>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<volatile double>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<volatile double>::denorm_min();

  std::numeric_limits<volatile long double>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<volatile long double>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<volatile long double>::denorm_min();

  std::numeric_limits<const volatile bool>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<const volatile bool>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<const volatile bool>::denorm_min();

  std::numeric_limits<const volatile int>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<const volatile int>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<const volatile int>::denorm_min();

  std::numeric_limits<const volatile float>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<const volatile float>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<const volatile float>::denorm_min();

  std::numeric_limits<const volatile double>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<const volatile double>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<const volatile double>::denorm_min();

  std::numeric_limits<const volatile long double>::has_denorm;      // expected-warning {{'has_denorm' is deprecated}}
  std::numeric_limits<const volatile long double>::has_denorm_loss; // expected-warning {{'has_denorm_loss' is deprecated}}
  std::numeric_limits<const volatile long double>::denorm_min();

  std::denorm_indeterminate; // expected-warning {{'denorm_indeterminate' is deprecated}}
  std::denorm_absent;        // expected-warning {{'denorm_absent' is deprecated}}
  std::denorm_present;       // expected-warning {{'denorm_present' is deprecated}}
}
