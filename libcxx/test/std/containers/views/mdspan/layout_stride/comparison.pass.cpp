//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template<class OtherMapping>
//   friend constexpr bool operator==(const mapping& x, const OtherMapping& y) noexcept;
//
// Constraints:
//   - layout-mapping-alike<OtherMapping> is satisfied.
//   - rank_ == OtherMapping::extents_type::rank() is true.
//   - OtherMapping::is_always_strided() is true.
//
// Preconditions: OtherMapping meets the layout mapping requirements ([mdspan.layout.policy.reqmts]).
//
// Returns: true if x.extents() == y.extents() is true, OFFSET(y) == 0 is true, and each of x.stride(r) == y.stride(r) is true for r in the range [0, x.extents().rank()). Otherwise, false.

#include <mdspan>
#include <type_traits>
#include <concepts>
#include <cassert>

#include "test_macros.h"

#include "../CustomTestLayouts.h"

template <class E1, class E2>
concept layout_mapping_comparable = requires(
    E1 e1,
    E2 e2,
    std::array<typename E1::index_type, E1::rank()> s1,
    std::array<typename E1::index_type, E1::rank()> s2) {
  std::layout_stride::mapping<E1>(e1, s1) == std::layout_stride::mapping<E2>(e2, s2);
};

template <class T1, class T2>
constexpr void test_comparison_different_rank() {
  constexpr size_t D = std::dynamic_extent;

  // sanity check same rank
  static_assert(layout_mapping_comparable<std::extents<T1, D>, std::extents<T2, D>>);
  static_assert(layout_mapping_comparable<std::extents<T1, 5>, std::extents<T2, D>>);
  static_assert(layout_mapping_comparable<std::extents<T1, D>, std::extents<T2, 5>>);
  static_assert(layout_mapping_comparable<std::extents<T1, 5>, std::extents<T2, 5>>);

  // not equality comparable when rank is not the same
  static_assert(!layout_mapping_comparable<std::extents<T1>, std::extents<T2, D>>);
  static_assert(!layout_mapping_comparable<std::extents<T1>, std::extents<T2, 1>>);
  static_assert(!layout_mapping_comparable<std::extents<T1, D>, std::extents<T2>>);
  static_assert(!layout_mapping_comparable<std::extents<T1, 1>, std::extents<T2>>);
  static_assert(!layout_mapping_comparable<std::extents<T1, D>, std::extents<T2, D, D>>);
  static_assert(!layout_mapping_comparable<std::extents<T1, 5>, std::extents<T2, 5, D>>);
  static_assert(!layout_mapping_comparable<std::extents<T1, 5>, std::extents<T2, 5, 1>>);
  static_assert(!layout_mapping_comparable<std::extents<T1, D, D>, std::extents<T2, D>>);
  static_assert(!layout_mapping_comparable<std::extents<T1, 5, D>, std::extents<T2, 5>>);
  static_assert(!layout_mapping_comparable<std::extents<T1, 5, 1>, std::extents<T2, 5>>);
}

template <class To, class From>
constexpr void test_comparison(
    bool equal,
    To dest_exts,
    From src_exts,
    std::array<int, To::rank()> dest_strides,
    std::array<int, From::rank()> src_strides) {
  std::layout_stride::mapping<To> dest(dest_exts, dest_strides);
  std::layout_stride::mapping<From> src(src_exts, src_strides);
  ASSERT_NOEXCEPT(dest == src);
  assert((dest == src) == equal);
  assert((dest != src) == !equal);
}

template <class T1, class T2>
constexpr void test_comparison_same_rank() {
  constexpr size_t D = std::dynamic_extent;

  test_comparison(true, std::extents<T1>(), std::extents<T2>(), std::array<int, 0>{}, std::array<int, 0>{});

  test_comparison(true, std::extents<T1, D>(5), std::extents<T2, D>(5), std::array<int, 1>{1}, std::array<int, 1>{1});
  test_comparison(true, std::extents<T1, D>(0), std::extents<T2, D>(0), std::array<int, 1>{1}, std::array<int, 1>{1});
  test_comparison(true, std::extents<T1, 5>(), std::extents<T2, D>(5), std::array<int, 1>{3}, std::array<int, 1>{3});
  test_comparison(true, std::extents<T1, D>(5), std::extents<T2, 5>(), std::array<int, 1>{1}, std::array<int, 1>{1});
  test_comparison(true, std::extents<T1, 5>(), std::extents< T2, 5>(), std::array<int, 1>{1}, std::array<int, 1>{1});
  test_comparison(false, std::extents<T1, 5>(), std::extents<T2, D>(5), std::array<int, 1>{2}, std::array<int, 1>{1});
  test_comparison(false, std::extents<T1, D>(5), std::extents<T2, D>(5), std::array<int, 1>{2}, std::array<int, 1>{1});
  test_comparison(false, std::extents<T1, D>(5), std::extents<T2, D>(7), std::array<int, 1>{1}, std::array<int, 1>{1});
  test_comparison(false, std::extents<T1, 5>(), std::extents<T2, D>(7), std::array<int, 1>{1}, std::array<int, 1>{1});
  test_comparison(false, std::extents<T1, D>(5), std::extents<T2, 7>(), std::array<int, 1>{1}, std::array<int, 1>{1});
  test_comparison(false, std::extents<T1, 5>(), std::extents<T2, 7>(), std::array<int, 1>{1}, std::array<int, 1>{1});

  test_comparison(
      true,
      std::extents<T1, D, D, D, D, D>(5, 6, 7, 8, 9),
      std::extents<T2, D, D, D, D, D>(5, 6, 7, 8, 9),
      std::array<int, 5>{2, 20, 200, 2000, 20000},
      std::array<int, 5>{2, 20, 200, 2000, 20000});
  test_comparison(
      true,
      std::extents<T1, D, 6, D, 8, D>(5, 7, 9),
      std::extents<T2, 5, D, D, 8, 9>(6, 7),
      std::array<int, 5>{2, 20, 200, 2000, 20000},
      std::array<int, 5>{2, 20, 200, 2000, 20000});
  test_comparison(
      true,
      std::extents<T1, 5, 6, 7, 8, 9>(5, 6, 7, 8, 9),
      std::extents<T2, 5, 6, 7, 8, 9>(),
      std::array<int, 5>{2, 20, 200, 2000, 20000},
      std::array<int, 5>{2, 20, 200, 2000, 20000});
  test_comparison(
      false,
      std::extents<T1, 5, 6, 7, 8, 9>(5, 6, 7, 8, 9),
      std::extents<T2, 5, 6, 7, 8, 9>(),
      std::array<int, 5>{2, 20, 200, 20000, 2000},
      std::array<int, 5>{2, 20, 200, 2000, 20000});
  test_comparison(
      false,
      std::extents<T1, D, D, D, D, D>(5, 6, 7, 8, 9),
      std::extents<T2, D, D, D, D, D>(5, 6, 3, 8, 9),
      std::array<int, 5>{2, 20, 200, 2000, 20000},
      std::array<int, 5>{2, 20, 200, 2000, 20000});
  test_comparison(
      false,
      std::extents<T1, D, 6, D, 8, D>(5, 7, 9),
      std::extents<T2, 5, D, D, 3, 9>(6, 7),
      std::array<int, 5>{2, 20, 200, 2000, 20000},
      std::array<int, 5>{2, 20, 200, 2000, 20000});
  test_comparison(
      false,
      std::extents<T1, 5, 6, 7, 8, 9>(5, 6, 7, 8, 9),
      std::extents<T2, 5, 6, 7, 3, 9>(),
      std::array<int, 5>{2, 20, 200, 2000, 20000},
      std::array<int, 5>{2, 20, 200, 2000, 20000});
}

template <class OtherLayout, class E1, class E2, class... OtherArgs>
constexpr void test_comparison_with(
    bool expect_equal, E1 e1, std::array<typename E1::index_type, E1::rank()> strides, E2 e2, OtherArgs... other_args) {
  typename std::layout_stride::template mapping<E1> map(e1, strides);
  typename OtherLayout::template mapping<E2> other_map(e2, other_args...);

  assert((map == other_map) == expect_equal);
}

template <class OtherLayout>
constexpr void test_comparison_with() {
  constexpr size_t D = std::dynamic_extent;
  bool is_left_based =
      std::is_same_v<OtherLayout, std::layout_left> || std::is_same_v<OtherLayout, always_convertible_layout>;
  test_comparison_with<OtherLayout>(true, std::extents<int>(), std::array<int, 0>{}, std::extents<unsigned>());
  test_comparison_with<OtherLayout>(true, std::extents<int, 5>(), std::array<int, 1>{1}, std::extents<unsigned, 5>());
  test_comparison_with<OtherLayout>(true, std::extents<int, D>(5), std::array<int, 1>{1}, std::extents<unsigned, 5>());
  test_comparison_with<OtherLayout>(false, std::extents<int, D>(5), std::array<int, 1>{2}, std::extents<unsigned, 5>());
  test_comparison_with<OtherLayout>(
      is_left_based, std::extents<int, D, D>(5, 7), std::array<int, 2>{1, 5}, std::extents<unsigned, D, D>(5, 7));
  test_comparison_with<OtherLayout>(
      !is_left_based, std::extents<int, D, D>(5, 7), std::array<int, 2>{7, 1}, std::extents<unsigned, D, D>(5, 7));
  test_comparison_with<OtherLayout>(
      false, std::extents<int, D, D>(5, 7), std::array<int, 2>{8, 1}, std::extents<unsigned, D, D>(5, 7));

  if constexpr (std::is_same_v<OtherLayout, always_convertible_layout>) {
    // test layout with strides not equal to product of extents
    test_comparison_with<OtherLayout>(
        true, std::extents<int, D, D>(5, 7), std::array<int, 2>{2, 10}, std::extents<unsigned, D, D>(5, 7), 0, 2);
    // make sure that offset != 0 results in false
    test_comparison_with<OtherLayout>(
        false, std::extents<int, D, D>(5, 7), std::array<int, 2>{2, 10}, std::extents<unsigned, D, D>(5, 7), 1, 2);
  }
}

template <class T1, class T2>
constexpr void test_comparison_index_type() {
  test_comparison_same_rank<T1, T2>();
  test_comparison_different_rank<T1, T2>();
  test_comparison_with<std::layout_right>();
  test_comparison_with<std::layout_left>();
  test_comparison_with<always_convertible_layout>();
}

constexpr bool test() {
  test_comparison_index_type<int, int>();
  test_comparison_index_type<int, size_t>();
  test_comparison_index_type<size_t, int>();
  test_comparison_index_type<size_t, long>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
