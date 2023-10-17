//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Test that entities declared [[nodiscard]] as an extension by libc++, are
// actually declared as such when _LIBCPP_DISABLE_NODISCARD_EXT is not specified.

// All entities to which libc++ applies [[nodiscard]] as an extension should
// be tested here and in nodiscard_extensions.pass.cpp. They should also
// be listed in `UsingLibcxx.rst` in the documentation for the extension.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <algorithm>
#include <bit> // bit_cast
#include <cstddef> // to_integer
#include <functional> // identity
#include <iterator>
#include <memory>
#include <utility> // to_underlying

#include "test_macros.h"

struct P {
  bool operator()(int) const { return false; }
};

void test_algorithms() {
  int arr[1] = { 1 };

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::adjacent_find(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::adjacent_find(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::all_of(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::any_of(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::binary_search(std::begin(arr), std::end(arr), 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::binary_search(std::begin(arr), std::end(arr), 1, std::greater<int>());

#if TEST_STD_VER >= 17
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::clamp(2, 1, 3);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::clamp(2, 1, 3, std::greater<int>());
#endif

#if TEST_STD_VER >= 23
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::contains(arr, arr + 1, 1);
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::count_if(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::count(std::begin(arr), std::end(arr), 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal_range(std::begin(arr), std::end(arr), 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal_range(std::begin(arr), std::end(arr), 1, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal(std::begin(arr), std::end(arr), std::begin(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal(std::begin(arr), std::end(arr), std::begin(arr),
             std::greater<int>());

#if TEST_STD_VER >= 14
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::equal(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
             std::greater<int>());
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_end(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_end(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
                std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_first_of(std::begin(arr), std::end(arr), std::begin(arr),
                     std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_first_of(std::begin(arr), std::end(arr), std::begin(arr),
                     std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_if_not(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find_if(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::find(std::begin(arr), std::end(arr), 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get_temporary_buffer<int>(1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::includes(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::includes(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
                std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_heap_until(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_heap_until(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_heap(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_heap(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_partitioned(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr),
                      std::greater<int>());

#if TEST_STD_VER >= 14
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr),
                      std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_permutation(std::begin(arr), std::end(arr), std::begin(arr),
                      std::end(arr), std::greater<int>());
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_sorted_until(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_sorted_until(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_sorted(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_sorted(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lexicographical_compare(std::begin(arr), std::end(arr), std::begin(arr),
                               std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lexicographical_compare(std::begin(arr), std::end(arr), std::begin(arr),
                               std::end(arr), std::greater<int>());

#if TEST_STD_VER >= 20
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lexicographical_compare_three_way(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lexicographical_compare_three_way(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr), std::compare_three_way());
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lower_bound(std::begin(arr), std::end(arr), 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lower_bound(std::begin(arr), std::end(arr), 1, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::max_element(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::max_element(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::max(1, 2);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::max(1, 2, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::max({1, 2, 3});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::max({1, 2, 3}, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min_element(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min_element(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min(1, 2);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min(1, 2, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min({1, 2, 3});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::min({1, 2, 3}, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::minmax_element(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::minmax_element(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::minmax(1, 2);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::minmax(1, 2, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::minmax({1, 2, 3});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::minmax({1, 2, 3}, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr),
                std::greater<int>());

#if TEST_STD_VER >= 14
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::mismatch(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
                std::greater<int>());
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::none_of(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::remove_if(std::begin(arr), std::end(arr), P());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::remove(std::begin(arr), std::end(arr), 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::search_n(std::begin(arr), std::end(arr), 1, 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::search_n(std::begin(arr), std::end(arr), 1, 1, std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::search(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::search(std::begin(arr), std::end(arr), std::begin(arr), std::end(arr),
              std::greater<int>());

#if TEST_STD_VER >= 17
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::search(std::begin(arr), std::end(arr),
              std::default_searcher(std::begin(arr), std::end(arr)));
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::unique(std::begin(arr), std::end(arr));

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::unique(std::begin(arr), std::end(arr), std::greater<int>());

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::upper_bound(std::begin(arr), std::end(arr), 1);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::upper_bound(std::begin(arr), std::end(arr), 1, std::greater<int>());
}

template<class LV, class RV>
void test_template_cast_wrappers(LV&& lv, RV&& rv) {
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::forward<LV>(lv);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::forward<RV>(rv);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(lv);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(rv);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move_if_noexcept(lv);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move_if_noexcept(rv);

#if TEST_STD_VER >= 17
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(lv);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(rv);
#endif

#if TEST_STD_VER >= 20
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::identity()(lv);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::identity()(rv);
#endif
}

void test_nontemplate_cast_wrappers()
{
#if TEST_STD_VER > 14
  std::byte b{42};
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::to_integer<int>(b);
#endif

#if TEST_STD_VER > 20
  enum E { Apple, Orange } e = Apple;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::to_underlying(e);
#endif
}

void f() {
  test_algorithms();

  int i = 42;
  test_template_cast_wrappers(i, std::move(i));
  test_nontemplate_cast_wrappers();
}
