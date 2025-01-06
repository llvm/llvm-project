//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template<class OtherExtents>
//   friend constexpr bool operator==(const mapping& x, const mapping<OtherExtents>& y) noexcept;
//                                      `
// Constraints: extents_type::rank() == OtherExtents::rank() is true.

#include <cassert>
#include <cstddef>
#include <mdspan>
#include <span> // dynamic_extent

#include "test_macros.h"

template <class To, class From>
constexpr void test_comparison(bool equal, To dest_exts, From src_exts) {
  std::layout_left::mapping<To> dest(dest_exts);
  std::layout_left::mapping<From> src(src_exts);
  ASSERT_NOEXCEPT(dest == src);
  assert((dest == src) == equal);
  assert((dest != src) == !equal);
}

struct X {
  constexpr bool does_not_match() { return true; }
};

constexpr X compare_layout_mappings(...) { return {}; }

template <class E1, class E2>
constexpr auto compare_layout_mappings(E1 e1, E2 e2)
    -> decltype(std::layout_left::mapping<E1>(e1) == std::layout_left::mapping<E2>(e2)) {
  return true;
}

template <class T1, class T2>
constexpr void test_comparison_different_rank() {
  constexpr size_t D = std::dynamic_extent;

  // sanity check same rank
  static_assert(compare_layout_mappings(std::extents<T1, D>(5), std::extents<T2, D>(5)));
  static_assert(compare_layout_mappings(std::extents<T1, 5>(), std::extents<T2, D>(5)));
  static_assert(compare_layout_mappings(std::extents<T1, D>(5), std::extents<T2, 5>()));
  static_assert(compare_layout_mappings(std::extents<T1, 5>(), std::extents<T2, 5>()));

  // not equality comparable when rank is not the same
  static_assert(compare_layout_mappings(std::extents<T1>(), std::extents<T2, D>(1)).does_not_match());
  static_assert(compare_layout_mappings(std::extents<T1>(), std::extents<T2, 1>()).does_not_match());

  static_assert(compare_layout_mappings(std::extents<T1, D>(1), std::extents<T2>()).does_not_match());
  static_assert(compare_layout_mappings(std::extents<T1, 1>(), std::extents<T2>()).does_not_match());

  static_assert(compare_layout_mappings(std::extents<T1, D>(5), std::extents<T2, D, D>(5, 5)).does_not_match());
  static_assert(compare_layout_mappings(std::extents<T1, 5>(), std::extents<T2, 5, D>(5)).does_not_match());
  static_assert(compare_layout_mappings(std::extents<T1, 5>(), std::extents<T2, 5, 1>()).does_not_match());

  static_assert(compare_layout_mappings(std::extents<T1, D, D>(5, 5), std::extents<T2, D>(5)).does_not_match());
  static_assert(compare_layout_mappings(std::extents<T1, 5, D>(5), std::extents<T2, D>(5)).does_not_match());
  static_assert(compare_layout_mappings(std::extents<T1, 5, 5>(), std::extents<T2, 5>()).does_not_match());
}

template <class T1, class T2>
constexpr void test_comparison_same_rank() {
  constexpr size_t D = std::dynamic_extent;

  test_comparison(true, std::extents<T1>(), std::extents<T2>());

  test_comparison(true, std::extents<T1, D>(5), std::extents<T2, D>(5));
  test_comparison(true, std::extents<T1, 5>(), std::extents<T2, D>(5));
  test_comparison(true, std::extents<T1, D>(5), std::extents<T2, 5>());
  test_comparison(true, std::extents<T1, 5>(), std::extents< T2, 5>());
  test_comparison(false, std::extents<T1, D>(5), std::extents<T2, D>(7));
  test_comparison(false, std::extents<T1, 5>(), std::extents<T2, D>(7));
  test_comparison(false, std::extents<T1, D>(5), std::extents<T2, 7>());
  test_comparison(false, std::extents<T1, 5>(), std::extents<T2, 7>());

  test_comparison(true, std::extents<T1, D, D, D, D, D>(5, 6, 7, 8, 9), std::extents<T2, D, D, D, D, D>(5, 6, 7, 8, 9));
  test_comparison(true, std::extents<T1, D, 6, D, 8, D>(5, 7, 9), std::extents<T2, 5, D, D, 8, 9>(6, 7));
  test_comparison(true, std::extents<T1, 5, 6, 7, 8, 9>(5, 6, 7, 8, 9), std::extents<T2, 5, 6, 7, 8, 9>());
  test_comparison(
      false, std::extents<T1, D, D, D, D, D>(5, 6, 7, 8, 9), std::extents<T2, D, D, D, D, D>(5, 6, 3, 8, 9));
  test_comparison(false, std::extents<T1, D, 6, D, 8, D>(5, 7, 9), std::extents<T2, 5, D, D, 3, 9>(6, 7));
  test_comparison(false, std::extents<T1, 5, 6, 7, 8, 9>(5, 6, 7, 8, 9), std::extents<T2, 5, 6, 7, 3, 9>());
}

template <class T1, class T2>
constexpr void test_comparison() {
  test_comparison_same_rank<T1, T2>();
  test_comparison_different_rank<T1, T2>();
}

constexpr bool test() {
  test_comparison<int, int>();
  test_comparison<int, size_t>();
  test_comparison<size_t, int>();
  test_comparison<size_t, long>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
