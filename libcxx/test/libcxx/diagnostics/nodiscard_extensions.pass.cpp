// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that entities declared [[nodiscard]] as at extension by libc++, are
// only actually declared such when _LIBCPP_ENABLE_NODISCARD is specified.

// This test intentionally leaks memory, so it is unsupported under ASAN.
// UNSUPPORTED: asan

// AppleClang9 and GCC 5 don't support C++17's implicitly synthesized
// deduction guides from existing ctors, needed by default_searcher() below.
// UNSUPPORTED: apple-clang-9
// UNSUPPORTED: gcc-5

// XFAIL: LIBCXX-WINDOWS-FIXME

// All entities to which libc++ applies [[nodiscard]] as an extension should
// be tested here and in nodiscard_extensions.fail.cpp. They should also
// be listed in `UsingLibcxx.rst` in the documentation for the extension.

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

  std::adjacent_find(std::begin(arr), std::end(arr));
  std::adjacent_find(std::begin(arr), std::end(arr), std::greater<int>());
  std::all_of(std::begin(arr), std::end(arr), P());
  std::any_of(std::begin(arr), std::end(arr), P());
  std::binary_search(std::begin(arr), std::end(arr), 1);
  std::binary_search(std::begin(arr), std::end(arr), 1, std::greater<int>());
#if TEST_STD_VER >= 17
  std::clamp(2, 1, 3);
  std::clamp(2, 1, 3, std::greater<int>());
#endif
  std::count_if(std::begin(arr), std::end(arr), P());
  std::count(std::begin(arr), std::end(arr), 1);
  std::equal_range(std::begin(arr), std::end(arr), 1);
  std::equal_range(std::begin(arr), std::end(arr), 1, std::greater<int>());
  std::equal(std::begin(arr), std::end(arr), std::begin(arr));
  std::equal(std::begin(arr), std::end(arr), std::begin(arr),
             std::greater<int>());
#if TEST_STD_VER >= 14
  std::equal(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));
  std::equal(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
             std::greater<int>());
#endif
  std::find_end(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));
  std::find_end(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
                std::greater<int>());
  std::find_first_of(std::begin(arr), std::end(arr), std::begin(arr),
                     std::end(arr));
  std::find_first_of(std::begin(arr), std::end(arr), std::begin(arr),
                     std::end(arr), std::greater<int>());
  std::find_if_not(std::begin(arr), std::end(arr), P());
  std::find_if(std::begin(arr), std::end(arr), P());
  std::find(std::begin(arr), std::end(arr), 1);
  std::get_temporary_buffer<int>(1); // intentional memory leak.
  std::includes(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));
  std::includes(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
                std::greater<int>());
  std::is_heap_until(std::begin(arr), std::end(arr));
  std::is_heap_until(std::begin(arr), std::end(arr), std::greater<int>());
  std::is_heap(std::begin(arr), std::end(arr));
  std::is_heap(std::begin(arr), std::end(arr), std::greater<int>());
  std::is_partitioned(std::begin(arr), std::end(arr), P());
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr));
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr),
                      std::greater<int>());
#if TEST_STD_VER >= 14
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr),
                      std::end(arr));
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr),
                      std::end(arr), std::greater<int>());
#endif
  std::is_sorted_until(std::begin(arr), std::end(arr));
  std::is_sorted_until(std::begin(arr), std::end(arr), std::greater<int>());
  std::is_sorted(std::begin(arr), std::end(arr));
  std::is_sorted(std::begin(arr), std::end(arr), std::greater<int>());
  std::lexicographical_compare(std::begin(arr), std::end(arr), std::begin(arr),
                               std::end(arr));
  std::lexicographical_compare(std::begin(arr), std::end(arr), std::begin(arr),
                               std::end(arr), std::greater<int>());
  std::lower_bound(std::begin(arr), std::end(arr), 1);
  std::lower_bound(std::begin(arr), std::end(arr), 1, std::greater<int>());
  std::max_element(std::begin(arr), std::end(arr));
  std::max_element(std::begin(arr), std::end(arr), std::greater<int>());
  std::max(1, 2);
  std::max(1, 2, std::greater<int>());
#if TEST_STD_VER >= 11
  std::max({1, 2, 3});
  std::max({1, 2, 3}, std::greater<int>());
#endif
  std::min_element(std::begin(arr), std::end(arr));
  std::min_element(std::begin(arr), std::end(arr), std::greater<int>());
  std::min(1, 2);
  std::min(1, 2, std::greater<int>());
#if TEST_STD_VER >= 11
  std::min({1, 2, 3});
  std::min({1, 2, 3}, std::greater<int>());
#endif
  std::minmax_element(std::begin(arr), std::end(arr));
  std::minmax_element(std::begin(arr), std::end(arr), std::greater<int>());
  std::minmax(1, 2);
  std::minmax(1, 2, std::greater<int>());
#if TEST_STD_VER >= 11
  std::minmax({1, 2, 3});
  std::minmax({1, 2, 3}, std::greater<int>());
#endif
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr));
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr),
                std::greater<int>());
#if TEST_STD_VER >= 14
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
                std::greater<int>());
#endif
  std::none_of(std::begin(arr), std::end(arr), P());
  std::remove_if(std::begin(arr), std::end(arr), P());
  std::remove(std::begin(arr), std::end(arr), 1);
  std::search_n(std::begin(arr), std::end(arr), 1, 1);
  std::search_n(std::begin(arr), std::end(arr), 1, 1, std::greater<int>());
  std::search(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));
  std::search(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
              std::greater<int>());
#if TEST_STD_VER >= 17
  std::search(std::begin(arr), std::end(arr),
              std::default_searcher(std::begin(arr), std::end(arr)));
#endif
  std::unique(std::begin(arr), std::end(arr));
  std::unique(std::begin(arr), std::end(arr), std::greater<int>());
  std::upper_bound(std::begin(arr), std::end(arr), 1);
  std::upper_bound(std::begin(arr), std::end(arr), 1, std::greater<int>());

  return 0;
}
