//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// constexpr mdspan(data_handle_type p, const mapping_type& m, const accessor_type& a);
//
// Preconditions: [0, m.required_span_size()) is an accessible range of p and a.
//
// Effects:
//   - Direct-non-list-initializes ptr_ with std::move(p),
//   - direct-non-list-initializes map_ with m, and
//   - direct-non-list-initializes acc_ with a.

#include <mdspan>
#include <cassert>
#include <concepts>
#include <span> // dynamic_extent
#include <type_traits>

#include "test_macros.h"

#include "../MinimalElementType.h"
#include "../CustomTestLayouts.h"
#include "CustomTestAccessors.h"

template <class H, class M, class A>
constexpr void test_mdspan_types(const H& handle, const M& map, const A& acc) {
  using MDS = std::mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  if (!std::is_constant_evaluated()) {
    move_counted_handle<typename MDS::element_type>::move_counter() = 0;
  }
  // use formulation of constructor which tests that it is not explicit
  MDS m = {handle, map, acc};
  if (!std::is_constant_evaluated()) {
    if constexpr (std::is_same_v<H, move_counted_handle<typename MDS::element_type>>) {
      assert((H::move_counter() == 1));
    }
  }
  LIBCPP_STATIC_ASSERT(!noexcept(MDS(handle, map, acc)));
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
  test_mdspan_types(handle, construct_mapping(layout, std::extents<signed char, D>(7)), acc);
  test_mdspan_types(handle, construct_mapping(layout, std::extents<unsigned, 7>()), acc);
  test_mdspan_types(handle, construct_mapping(layout, std::extents<size_t, D, 4, D>(2, 3)), acc);
  test_mdspan_types(handle, construct_mapping(layout, std::extents<signed char, D, 7, D>(0, 3)), acc);
  test_mdspan_types(handle, construct_mapping(layout, std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc);
}

template <class H, class A>
constexpr void mixin_layout(const H& handle, const A& acc) {
  mixin_extents(handle, std::layout_left(), acc);
  mixin_extents(handle, std::layout_right(), acc);
  mixin_extents(handle, layout_wrapping_integral<4>(), acc);
}

template <class T>
constexpr void mixin_accessor() {
  ElementPool<T, 1024> elements;
  mixin_layout(elements.get_ptr(), std::default_accessor<T>());

  // Using weird accessor/data_handle
  // Make sure they actually got the properties we want to test
  // checked_accessor is not default constructible except for const double, where it is not noexcept
  static_assert(std::is_default_constructible_v<checked_accessor<T>> == std::is_same_v<T, const double>);
  // checked_accessor's data handle type is not default constructible for double
  static_assert(
      std::is_default_constructible_v<typename checked_accessor<T>::data_handle_type> != std::is_same_v<T, double>);
  mixin_layout(typename checked_accessor<T>::data_handle_type(elements.get_ptr()), checked_accessor<T>(1024));
}

template <class E>
using mapping_t = std::layout_right::mapping<E>;

constexpr bool test() {
  mixin_accessor<int>();
  mixin_accessor<const int>();
  mixin_accessor<double>();
  mixin_accessor<const double>();
  mixin_accessor<MinimalElementType>();
  mixin_accessor<const MinimalElementType>();

  // test non-constructibility from wrong args
  constexpr size_t D = std::dynamic_extent;
  using mds_t        = std::mdspan<float, std::extents<int, 3, D, D>>;
  using acc_t        = std::default_accessor<float>;

  // sanity check
  static_assert(std::is_constructible_v<mds_t, float*, mapping_t<std::extents<int, 3, D, D>>, acc_t>);

  // test non-constructibility from wrong accessor
  static_assert(
      !std::
          is_constructible_v<mds_t, float*, mapping_t<std::extents<int, 3, D, D>>, std::default_accessor<const float>>);

  // test non-constructibility from wrong mapping type
  // wrong rank
  static_assert(!std::is_constructible_v<mds_t, float*, mapping_t<std::extents<int, D, D>>, acc_t>);
  static_assert(!std::is_constructible_v<mds_t, float*, mapping_t<std::extents<int, D, D, D, D>>, acc_t>);
  // wrong type in general: note the map constructor does NOT convert, since it takes by const&
  static_assert(!std::is_constructible_v<mds_t, float*, mapping_t<std::extents<int, D, D, D>>, acc_t>);
  static_assert(!std::is_constructible_v<mds_t, float*, mapping_t<std::extents<unsigned, 3, D, D>>, acc_t>);

  // test non-constructibility from wrong handle_type
  static_assert(!std::is_constructible_v<mds_t, const float*, mapping_t<std::extents<int, 3, D, D>>, acc_t>);

  return true;
}
int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
