//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Check that functions are marked [[nodiscard]]

#include <numeric>

#include "test_macros.h"

void test() {
  {
    std::initializer_list<int> il{94, 82};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    accumulate(il.begin(), il.end(), 49);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    accumulate(il.begin(), il.end(), 49, std::multiplies<>());

  }

#if TEST_STD_VER >= 26
  // [numeric.sat]
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::saturating_add(94, 82);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::saturating_sub(94, 82);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::saturating_mul(94, 82);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::saturating_div(94, 82);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::saturating_cast<signed int>(49);
#endif // TEST_STD_VER >= 26

#if TEST_STD_VER >= 20
  {
    int arr[]{94, 82, 49};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::midpoint(94, 82);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::midpoint(arr, arr + 2);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::midpoint(94.0, 82.0);
  }
#endif
}
