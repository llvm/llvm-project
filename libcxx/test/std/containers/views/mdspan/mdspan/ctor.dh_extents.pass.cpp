//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// constexpr mdspan(data_handle_type p, const extents_type& ext);
//
// Constraints:
//   - is_constructible_v<mapping_type, const extents_type&> is true, and
//   - is_default_constructible_v<accessor_type> is true.
//
// Preconditions: [0, map_.required_span_size()) is an accessible range of p and acc_
//                for the values of map_ and acc_ after the invocation of this constructor.
//
// Effects:
//   - Direct-non-list-initializes ptr_ with std::move(p),
//   - direct-non-list-initializes map_ with ext, and
//   - value-initializes acc_.

#include <mdspan>
#include <type_traits>
#include <concepts>
#include <cassert>

#include "test_macros.h"

#include "../MinimalElementType.h"
#include "../CustomTestLayouts.h"
#include "CustomTestAccessors.h"

template <bool mec, bool ac, class H, class M, class A>
constexpr void test_mdspan_types(const H& handle, const M& map, const A&) {
  using MDS = std::mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  static_assert(mec == std::is_constructible_v<M, const typename M::extents_type&>);
  static_assert(ac == std::is_default_constructible_v<A>);
  if constexpr (mec && ac) {
    if (!std::is_constant_evaluated()) {
      move_counted_handle<typename MDS::element_type>::move_counter() = 0;
    }
    // use formulation of constructor which tests that its not explicit
    MDS m = {handle, map.extents()};
    if (!std::is_constant_evaluated()) {
      if constexpr (std::is_same_v<H, move_counted_handle<typename MDS::element_type>>) {
        assert((H::move_counter() == 1));
      }
    }
    LIBCPP_STATIC_ASSERT(!noexcept(MDS(handle, map.extents())));
    assert(m.extents() == map.extents());
    if constexpr (std::equality_comparable<H>)
      assert(m.data_handle() == handle);
    if constexpr (std::equality_comparable<M>)
      assert(m.mapping() == map);
    if constexpr (std::equality_comparable<A>)
      assert(m.accessor() == A());
  } else {
    static_assert(!std::is_constructible_v<MDS, const H&, const typename M::extents_type&>);
  }
}

template <bool mec, bool ac, class H, class L, class A>
constexpr void mixin_extents(const H& handle, const L& layout, const A& acc) {
  constexpr size_t D = std::dynamic_extent;
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, std::extents<int>()), acc);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, std::extents<signed char, D>(7)), acc);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, std::extents<unsigned, 7>()), acc);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, std::extents<size_t, D, 4, D>(2, 3)), acc);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, std::extents<signed char, D, 7, D>(0, 3)), acc);
  test_mdspan_types<mec, ac>(
      handle, construct_mapping(layout, std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc);
}

template <bool ac, class H, class A>
constexpr void mixin_layout(const H& handle, const A& acc) {
  mixin_extents<true, ac>(handle, std::layout_left(), acc);
  mixin_extents<true, ac>(handle, std::layout_right(), acc);

  // Use weird layout, make sure it has the properties we want to test
  // Sanity check that this layouts mapping is constructible from extents (via its move constructor)
  static_assert(std::is_constructible_v<layout_wrapping_integral<8>::mapping<std::extents<int>>, std::extents<int>>);
  static_assert(
      !std::is_constructible_v<layout_wrapping_integral<8>::mapping<std::extents<int>>, const std::extents<int>&>);
  mixin_extents<false, ac>(handle, layout_wrapping_integral<8>(), acc);
  // Sanity check that this layouts mapping is not constructible from extents
  static_assert(!std::is_constructible_v<layout_wrapping_integral<4>::mapping<std::extents<int>>, std::extents<int>>);
  static_assert(
      !std::is_constructible_v<layout_wrapping_integral<4>::mapping<std::extents<int>>, const std::extents<int>&>);
  mixin_extents<false, ac>(handle, layout_wrapping_integral<4>(), acc);
}

template <class T>
constexpr void mixin_accessor() {
  ElementPool<T, 1024> elements;
  mixin_layout<true>(elements.get_ptr(), std::default_accessor<T>());

  // Using weird accessor/data_handle
  // Make sure they actually got the properties we want to test
  // checked_accessor is not default constructible except for const double, where it is not noexcept
  static_assert(std::is_default_constructible_v<checked_accessor<T>> == std::is_same_v<T, const double>);
  mixin_layout<std::is_same_v<T, const double>>(
      typename checked_accessor<T>::data_handle_type(elements.get_ptr()), checked_accessor<T>(1024));
}

constexpr bool test() {
  mixin_accessor<int>();
  mixin_accessor<const int>();
  mixin_accessor<double>();
  mixin_accessor<const double>();
  mixin_accessor<MinimalElementType>();
  mixin_accessor<const MinimalElementType>();

  // test non-constructibility from wrong extents type
  constexpr size_t D = std::dynamic_extent;
  using mds_t        = std::mdspan<float, std::extents<int, 3, D, D>>;
  // sanity check
  static_assert(std::is_constructible_v<mds_t, float*, std::extents<int, 3, D, D>>);
  // wrong size
  static_assert(!std::is_constructible_v<mds_t, float*, std::extents<int, D, D>>);
  static_assert(!std::is_constructible_v<mds_t, float*, std::extents<int, D, D, D, D>>);
  // wrong type in general: note the extents constructor does NOT convert, since it takes by const&
  static_assert(!std::is_constructible_v<mds_t, float*, std::extents<int, D, D, D>>);
  static_assert(!std::is_constructible_v<mds_t, float*, std::extents<unsigned, 3, D, D>>);

  // test non-constructibility from wrong handle_type
  static_assert(!std::is_constructible_v<mds_t, const float*, std::extents<int, 3, D, D>>);

  return true;
}
int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
