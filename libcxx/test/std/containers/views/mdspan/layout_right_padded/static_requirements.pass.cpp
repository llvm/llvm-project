//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <mdspan>

#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <type_traits>
#include <utility>

#include "test_macros.h"

template <class Mapping, size_t... Indices>
void test_mapping_requirements(std::index_sequence<Indices...>) {
  using Extents = Mapping::extents_type;

  LIBCPP_STATIC_ASSERT(std::__mdspan_detail::__is_extents_v<Extents>);
  static_assert(std::is_copy_constructible_v<Mapping>);
  static_assert(std::is_nothrow_move_constructible_v<Mapping>);
  static_assert(std::is_nothrow_move_assignable_v<Mapping>);
  static_assert(std::is_nothrow_swappable_v<Mapping>);

  ASSERT_SAME_TYPE(typename Mapping::index_type, typename Extents::index_type);
  ASSERT_SAME_TYPE(typename Mapping::size_type, typename Extents::size_type);
  ASSERT_SAME_TYPE(typename Mapping::rank_type, typename Extents::rank_type);
  ASSERT_SAME_TYPE(typename Mapping::layout_type, std::layout_right_padded<Mapping::padding_value>);
  ASSERT_SAME_TYPE(typename Mapping::layout_type::template mapping<Extents>, Mapping);
  static_assert(std::is_same_v<decltype(std::declval<Mapping>().extents()), const Extents&>);
  static_assert(std::is_same_v<decltype(std::declval<Mapping>().strides()),
                               std::array<typename Mapping::index_type, Extents::rank()>>);
  static_assert(std::is_same_v<decltype(std::declval<Mapping>()(Indices...)), typename Mapping::index_type>);
  static_assert(std::is_same_v<decltype(std::declval<Mapping>().required_span_size()), typename Mapping::index_type>);
  static_assert(std::is_same_v<decltype(std::declval<Mapping>().is_unique()), bool>);
  static_assert(std::is_same_v<decltype(std::declval<Mapping>().is_exhaustive()), bool>);
  static_assert(std::is_same_v<decltype(std::declval<Mapping>().is_strided()), bool>);
  if constexpr (Extents::rank() > 0)
    static_assert(std::is_same_v<decltype(std::declval<Mapping>().stride(0)), typename Mapping::index_type>);
  static_assert(std::is_same_v<decltype(Mapping::is_always_unique()), bool>);
  static_assert(std::is_same_v<decltype(Mapping::is_always_exhaustive()), bool>);
  static_assert(std::is_same_v<decltype(Mapping::is_always_strided()), bool>);
}

template <class Layout, class Extents>
void test_layout_mapping_requirements() {
  test_mapping_requirements<typename Layout::template mapping<Extents>>(std::make_index_sequence<Extents::rank()>());
}

int main() {
  constexpr size_t D = std::dynamic_extent;

  test_layout_mapping_requirements<std::layout_right_padded<4>, std::extents<int8_t>>();
  test_layout_mapping_requirements<std::layout_right_padded<4>, std::extents<uint8_t, 4, 6>>();
  test_layout_mapping_requirements<std::layout_right_padded<4>, std::extents<int32_t, D, 4>>();
  test_layout_mapping_requirements<std::layout_right_padded<4>, std::extents<uint32_t, D, D>>();

  test_layout_mapping_requirements<std::layout_right_padded<D>, std::extents<int8_t, 1, D, D>>();
  test_layout_mapping_requirements<std::layout_right_padded<D>, std::extents<uint8_t, D, D, D>>();
  test_layout_mapping_requirements<std::layout_right_padded<D>, std::extents<int32_t, D, D, D>>();
  test_layout_mapping_requirements<std::layout_right_padded<D>, std::extents<uint32_t, D, D, D>>();
  return 0;
}
