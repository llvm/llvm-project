//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// Check that PSTL algorithms are marked [[nodiscard]]

#include <algorithm>
#include <execution>
#include <iterator>
#include <numeric>

void test() {
  int a[]    = {1};
  int b[]    = {1};
  auto pred  = [](auto) { return false; };
  auto pred2 = [](auto, auto) { return false; };

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::all_of(std::execution::par, std::begin(a), std::end(a), pred);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::any_of(std::execution::par, std::begin(a), std::end(a), pred);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::none_of(std::execution::par, std::begin(a), std::end(a), pred);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::count_if(std::execution::par, std::begin(a), std::end(a), pred);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::count(std::execution::par, std::begin(a), std::end(a), 1);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal(std::execution::par, std::begin(a), std::end(a), std::begin(b), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal(std::execution::par, std::begin(a), std::end(a), std::begin(b));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal(std::execution::par, std::begin(a), std::end(a), std::begin(b), std::end(b), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal(std::execution::par, std::begin(a), std::end(a), std::begin(b), std::end(b));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_end(std::execution::par, std::begin(a), std::end(a), std::begin(b), std::end(b));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_end(std::execution::par, std::begin(a), std::end(a), std::begin(b), std::end(b), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_if(std::execution::par, std::begin(a), std::end(a), pred);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_if_not(std::execution::par, std::begin(a), std::end(a), pred);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find(std::execution::par, std::begin(a), std::end(a), 1);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_first_of(std::execution::par, std::begin(a), std::end(a), std::begin(b), std::end(b));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_first_of(std::execution::par, std::begin(a), std::end(a), std::begin(b), std::end(b), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_heap_until(std::execution::par, std::begin(a), std::end(a));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_heap_until(std::execution::par, std::begin(a), std::end(a), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_partitioned(std::execution::par, std::begin(a), std::end(a), pred);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_sorted(std::execution::par, std::begin(a), std::end(a));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_sorted(std::execution::par, std::begin(a), std::end(a), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_sorted_until(std::execution::par, std::begin(a), std::end(a));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_sorted_until(std::execution::par, std::begin(a), std::end(a), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::reverse_copy(std::execution::par, std::begin(a), std::end(a), std::begin(b));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::adjacent_difference(std::execution::par, std::begin(a), std::end(a), std::begin(b));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::adjacent_difference(std::execution::par, std::begin(a), std::end(a), std::begin(b), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::max_element(std::execution::par, std::begin(a), std::end(a));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::max_element(std::execution::par, std::begin(a), std::end(a), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min_element(std::execution::par, std::begin(a), std::end(a));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min_element(std::execution::par, std::begin(a), std::end(a), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::mismatch(std::execution::par, std::begin(a), std::end(a), std::begin(b));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::mismatch(std::execution::par, std::begin(a), std::end(a), std::begin(b), std::end(b));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::mismatch(std::execution::par, std::begin(a), std::end(a), std::begin(b), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::mismatch(std::execution::par, std::begin(a), std::end(a), std::begin(b), std::end(b), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::adjacent_find(std::execution::par, std::begin(a), std::end(a));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::adjacent_find(std::execution::par, std::begin(a), std::end(a), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lexicographical_compare(std::execution::par, std::begin(a), std::end(a), std::begin(b), std::end(b));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lexicographical_compare(std::execution::par, std::begin(a), std::end(a), std::begin(b), std::end(b), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::search(std::execution::par, std::begin(a), std::end(a), std::begin(b), std::end(b));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::search(std::execution::par, std::begin(a), std::end(a), std::begin(b), std::end(b), pred2);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::search_n(std::execution::par, std::begin(a), std::end(a), 1, 1);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::search_n(std::execution::par, std::begin(a), std::end(a), 1, 1, pred2);
}
