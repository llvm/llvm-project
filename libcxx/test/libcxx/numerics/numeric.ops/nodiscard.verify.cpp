//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// <numeric>

// Check that functions are marked [[nodiscard]]

#include <execution>
#include <functional>
#include <numeric>
#include <vector>

#include "test_macros.h"

void test() {
  {
    std::vector<int> vec(94, 82);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::accumulate(vec.begin(), vec.end(), 49);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::accumulate(vec.begin(), vec.end(), 49, std::multiplies<int>());
  }

  {
    std::vector<int> vec(94, 82);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::inner_product(vec.begin(), vec.end(), vec.begin(), 49);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::inner_product(vec.begin(), vec.end(), vec.begin(), 49, std::multiplies<int>(), std::plus<int>());
  }

#if TEST_STD_VER >= 17
  {
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::gcd(94, 82);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::lcm(94, 82);
  }
#endif // TEST_STD_VER >= 17

#if TEST_STD_VER >= 17
  {
    std::initializer_list<int> il{94, 82};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::reduce(std::execution::seq, il.begin(), il.end(), 49);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::reduce(std::execution::seq, il.begin(), il.end(), 49, std::multiplies<int>());
  }

  {
    std::initializer_list<int> il{94, 82};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::reduce(il.begin(), il.end(), 49);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::reduce(il.begin(), il.end(), 49, std::multiplies<int>());
  }
#endif // TEST_STD_VER >= 17

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

#if TEST_STD_VER >= 17
  {
    std::initializer_list<int> il{94, 82};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::reduce(std::execution::par, il.begin(), il.end(), 49);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::reduce(std::execution::par, il.begin(), il.end(), 49, std::multiplies<int>());
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::reduce(std::execution::par_unseq, il.begin(), il.end(), 49);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::reduce(std::execution::par_unseq, il.begin(), il.end(), 49, std::multiplies<int>());
  }

  {
    std::initializer_list<int> il{94, 82};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::reduce(il.begin(), il.end(), 49);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::reduce(il.begin(), il.end(), 49, std::multiplies<int>());
  }
#endif

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

#if TEST_STD_VER >= 17
  {
    std::initializer_list<int> il{94, 82};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::transform_reduce(il.begin(), il.end(), 49, std::plus<int>(), std::negate<int>());
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::transform_reduce(il.begin(), il.end(), il.begin(), 49, std::plus<int>(), std::multiplies<int>());
  }
#endif // TEST_STD_VER >= 17
}
