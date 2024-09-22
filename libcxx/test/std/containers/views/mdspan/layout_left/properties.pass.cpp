//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// namespace std {
//   template<class Extents>
//   class layout_left::mapping {
//
//     ...
//     static constexpr bool is_always_unique() noexcept { return true; }
//     static constexpr bool is_always_exhaustive() noexcept { return true; }
//     static constexpr bool is_always_strided() noexcept { return true; }
//
//     static constexpr bool is_unique() noexcept { return true; }
//     static constexpr bool is_exhaustive() noexcept { return true; }
//     static constexpr bool is_strided() noexcept { return true; }
//     ...
//   };
// }

#include <mdspan>
#include <cassert>
#include <concepts>
#include <span> // dynamic_extent
#include <type_traits>
#include <utility>

#include "test_macros.h"

template <class E>
constexpr void test_layout_mapping_left() {
  using M = std::layout_left::mapping<E>;
  assert(M::is_unique() == true);
  assert(M::is_exhaustive() == true);
  assert(M::is_strided() == true);
  assert(M::is_always_unique() == true);
  assert(M::is_always_exhaustive() == true);
  assert(M::is_always_strided() == true);
  ASSERT_NOEXCEPT(std::declval<M>().is_unique());
  ASSERT_NOEXCEPT(std::declval<M>().is_exhaustive());
  ASSERT_NOEXCEPT(std::declval<M>().is_strided());
  ASSERT_NOEXCEPT(M::is_always_unique());
  ASSERT_NOEXCEPT(M::is_always_exhaustive());
  ASSERT_NOEXCEPT(M::is_always_strided());
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;
  test_layout_mapping_left<std::extents<int>>();
  test_layout_mapping_left<std::extents<signed char, 4, 5>>();
  test_layout_mapping_left<std::extents<unsigned, D, 4>>();
  test_layout_mapping_left<std::extents<size_t, D, D, D, D>>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
