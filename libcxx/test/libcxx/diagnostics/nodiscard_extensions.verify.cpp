// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// AppleClang9 doesn't yet support C++17's implicitly synthesized deduction
// guides from existing ctors, needed by default_searcher() below.
// UNSUPPORTED: apple-clang-9

// Test that entities declared [[nodiscard]] as an extension by libc++, are
// only actually declared such when _LIBCPP_ENABLE_NODISCARD is specified.

// All entities to which libc++ applies [[nodiscard]] as an extension should
// be tested here and in nodiscard_extensions.pass.cpp. They should also
// be listed in `UsingLibcxx.rst` in the documentation for the extension.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_NODISCARD

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>

#include "test_macros.h"

struct P {
  bool operator()(int) const { return false; }
};

int main(int, char**) {
  int arr[1] = { 1 };

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::adjacent_find(std::begin(arr), std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::adjacent_find(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::all_of(std::begin(arr), std::end(arr), P());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::any_of(std::begin(arr), std::end(arr), P());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::binary_search(std::begin(arr), std::end(arr), 1);

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::binary_search(std::begin(arr), std::end(arr), 1, std::greater<int>());

#if TEST_STD_VER >= 17
  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::clamp(2, 1, 3);

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::clamp(2, 1, 3, std::greater<int>());
#endif

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::count_if(std::begin(arr), std::end(arr), P());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::count(std::begin(arr), std::end(arr), 1);

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::equal_range(std::begin(arr), std::end(arr), 1);

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::equal_range(std::begin(arr), std::end(arr), 1, std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::equal(std::begin(arr), std::end(arr), std::begin(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::equal(std::begin(arr), std::end(arr), std::begin(arr),
             std::greater<int>());

#if TEST_STD_VER >= 14
  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::equal(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::equal(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
             std::greater<int>());
#endif

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::find_end(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::find_end(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
                std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::find_first_of(std::begin(arr), std::end(arr), std::begin(arr),
                     std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::find_first_of(std::begin(arr), std::end(arr), std::begin(arr),
                     std::end(arr), std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::find_if_not(std::begin(arr), std::end(arr), P());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::find_if(std::begin(arr), std::end(arr), P());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::find(std::begin(arr), std::end(arr), 1);

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::get_temporary_buffer<int>(1);

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::includes(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::includes(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
                std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::is_heap_until(std::begin(arr), std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::is_heap_until(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::is_heap(std::begin(arr), std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::is_heap(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::is_partitioned(std::begin(arr), std::end(arr), P());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr),
                      std::greater<int>());

#if TEST_STD_VER >= 14
  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr),
                      std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr),
                      std::end(arr), std::greater<int>());
#endif

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::is_sorted_until(std::begin(arr), std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::is_sorted_until(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::is_sorted(std::begin(arr), std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::is_sorted(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::lexicographical_compare(std::begin(arr), std::end(arr), std::begin(arr),
                               std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::lexicographical_compare(std::begin(arr), std::end(arr), std::begin(arr),
                               std::end(arr), std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::lower_bound(std::begin(arr), std::end(arr), 1);

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::lower_bound(std::begin(arr), std::end(arr), 1, std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::max_element(std::begin(arr), std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::max_element(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::max(1, 2);

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::max(1, 2, std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::max({1, 2, 3});

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::max({1, 2, 3}, std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::min_element(std::begin(arr), std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::min_element(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::min(1, 2);

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::min(1, 2, std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::min({1, 2, 3});

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::min({1, 2, 3}, std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::minmax_element(std::begin(arr), std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::minmax_element(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::minmax(1, 2);

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::minmax(1, 2, std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::minmax({1, 2, 3});

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::minmax({1, 2, 3}, std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr),
                std::greater<int>());

#if TEST_STD_VER >= 14
  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
                std::greater<int>());
#endif

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::none_of(std::begin(arr), std::end(arr), P());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::remove_if(std::begin(arr), std::end(arr), P());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::remove(std::begin(arr), std::end(arr), 1);

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::search_n(std::begin(arr), std::end(arr), 1, 1);

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::search_n(std::begin(arr), std::end(arr), 1, 1, std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::search(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::search(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
              std::greater<int>());

#if TEST_STD_VER >= 17
  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::search(std::begin(arr), std::end(arr),
              std::default_searcher(std::begin(arr), std::end(arr)));
#endif

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::unique(std::begin(arr), std::end(arr));

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::unique(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::upper_bound(std::begin(arr), std::end(arr), 1);

  // expected-warning-re@+1 {{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  std::upper_bound(std::begin(arr), std::end(arr), 1, std::greater<int>());

  return 0;
}
