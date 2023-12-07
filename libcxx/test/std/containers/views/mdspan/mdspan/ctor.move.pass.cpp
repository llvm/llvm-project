//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// constexpr mdspan(mdspan&&) = default;
//
// A specialization of mdspan is a trivially copyable type if its accessor_type, mapping_type, and data_handle_type are trivially copyable types.

#include <mdspan>
#include <type_traits>
#include <concepts>
#include <cassert>

#include "test_macros.h"

#include "../MinimalElementType.h"
#include "CustomTestLayouts.h"
#include "CustomTestAccessors.h"

template <class H, class M, class A>
constexpr void test_mdspan_types(const H& handle, const M& map, const A& acc) {
  using MDS = std::mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  MDS m_org(handle, map, acc);
  MDS m(std::move(m_org));
  static_assert(std::is_trivially_move_constructible_v<MDS> ==
                (std::is_trivially_move_constructible_v<H> && std::is_trivially_move_constructible_v<M> &&
                 std::is_trivially_move_constructible_v<A>));
  assert(m.extents() == map.extents());
  if constexpr (std::equality_comparable<H>)
    assert(m.data_handle() == handle);
  if constexpr (std::equality_comparable<M>)
    assert(m.mapping() == map);
  if constexpr (std::equality_comparable<A>)
    assert(m.accessor() == acc);
}

template <class H, class L, class A>
constexpr void mixin_extents(const H& handle, const L& layout, const A& acc) {
  constexpr size_t D = std::dynamic_extent;
  test_mdspan_types(handle, construct_mapping(layout, std::extents<int>()), acc);
  test_mdspan_types(handle, construct_mapping(layout, std::extents<char, D>(7)), acc);
  test_mdspan_types(handle, construct_mapping(layout, std::extents<unsigned, 7>()), acc);
  test_mdspan_types(handle, construct_mapping(layout, std::extents<size_t, D, 4, D>(2, 3)), acc);
  test_mdspan_types(handle, construct_mapping(layout, std::extents<char, D, 7, D>(0, 3)), acc);
  test_mdspan_types(handle, construct_mapping(layout, std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc);
}

template <class H, class A>
constexpr void mixin_layout(const H& handle, const A& acc) {
  // make sure we test a trivially copyable mapping
  static_assert(std::is_trivially_move_constructible_v<typename std::layout_left::template mapping<std::extents<int>>>);
  mixin_extents(handle, std::layout_left(), acc);
  mixin_extents(handle, std::layout_right(), acc);
  // make sure we test a not trivially copyable mapping
  static_assert(!std::is_trivially_move_constructible_v<
                typename layout_wrapping_integral<4>::template mapping<std::extents<int>>>);
  mixin_extents(handle, layout_wrapping_integral<4>(), acc);
}

template <class T>
constexpr void mixin_accessor() {
  ElementPool<T, 1024> elements;
  // make sure we test trivially constructible accessor and data_handle
  static_assert(std::is_trivially_move_constructible_v<std::default_accessor<T>>);
  static_assert(std::is_trivially_move_constructible_v<typename std::default_accessor<T>::data_handle_type>);
  mixin_layout(elements.get_ptr(), std::default_accessor<T>());

  // Using weird accessor/data_handle
  // Make sure they actually got the properties we want to test
  // checked_accessor is noexcept copy constructible except for const double
  checked_accessor<T> acc(1024);
  static_assert(std::is_trivially_move_constructible_v<typename checked_accessor<T>::data_handle_type> ==
                std::is_same_v<T, double>);
  mixin_layout(typename checked_accessor<T>::data_handle_type(elements.get_ptr()), acc);
}

constexpr bool test() {
  mixin_accessor<int>();
  mixin_accessor<const int>();
  mixin_accessor<double>();
  mixin_accessor<const double>();
  mixin_accessor<MinimalElementType>();
  mixin_accessor<const MinimalElementType>();
  return true;
}
int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
