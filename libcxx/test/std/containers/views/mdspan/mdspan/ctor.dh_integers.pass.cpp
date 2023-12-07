//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template<class... OtherIndexTypes>
//   constexpr explicit mdspan(data_handle_type p, OtherIndexTypes... exts);
//
// Let N be sizeof...(OtherIndexTypes).
//
// Constraints:
//   - (is_convertible_v<OtherIndexTypes, index_type> && ...) is true,
//   - (is_nothrow_constructible<index_type, OtherIndexTypes> && ...) is true,
//   - N == rank() || N == rank_dynamic() is true,
//   - is_constructible_v<mapping_type, extents_type> is true, and
//   - is_default_constructible_v<accessor_type> is true.
//
// Preconditions: [0, map_.required_span_size()) is an accessible range of p and acc_
//                for the values of map_ and acc_ after the invocation of this constructor.
//
// Effects:
//   - Direct-non-list-initializes ptr_ with std::move(p),
//   - direct-non-list-initializes map_ with extents_type(static_cast<index_type>(std::move(exts))...), and
//   - value-initializes acc_.

#include <array>
#include <concepts>
#include <cassert>
#include <mdspan>
#include <type_traits>

#include "test_macros.h"

#include "../MinimalElementType.h"
#include "CustomTestLayouts.h"
#include "CustomTestAccessors.h"

template <class MDS, class... Args>
concept check_mdspan_ctor_implicit = requires(MDS m, Args... args) { m = {args...}; };

template <bool mec, bool ac, class H, class M, class A, class... Idxs>
constexpr void test_mdspan_types(const H& handle, const M& map, const A&, Idxs... idxs) {
  using MDS = std::mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;

  static_assert(mec == std::is_constructible_v<M, typename M::extents_type>);
  static_assert(ac == std::is_default_constructible_v<A>);

  if constexpr (mec && ac) {
    if !consteval {
      move_counted_handle<typename MDS::element_type>::move_counter() = 0;
    }
    MDS m(handle, idxs...);
    if !consteval {
      if constexpr (std::is_same_v<H, move_counted_handle<typename MDS::element_type>>) {
        assert((H::move_counter() == 1));
      }
    }

    // sanity check that concept works
    static_assert(check_mdspan_ctor_implicit<MDS, H, std::array<typename MDS::index_type, MDS::rank_dynamic()>>);
    // check that the constructor from integral is explicit
    static_assert(!check_mdspan_ctor_implicit<MDS, H, Idxs...>);

    assert(m.extents() == map.extents());
    if constexpr (std::equality_comparable<H>)
      assert(m.data_handle() == handle);
    if constexpr (std::equality_comparable<M>)
      assert(m.mapping() == map);
    if constexpr (std::equality_comparable<A>)
      assert(m.accessor() == A());
  } else {
    static_assert(!std::is_constructible_v<MDS, const H&, Idxs... >);
  }
}

template <bool mec, bool ac, class H, class L, class A>
constexpr void mixin_extents(const H& handle, const L& layout, const A& acc) {
  constexpr size_t D = std::dynamic_extent;
  // construct from just dynamic extents
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, std::extents<int>()), acc);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, std::extents<char, D>(7)), acc, 7);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, std::extents<unsigned, 7>()), acc);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, std::extents<size_t, D, 4, D>(2, 3)), acc, 2, 3);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, std::extents<char, D, 7, D>(0, 3)), acc, 0, 3);
  test_mdspan_types<mec, ac>(
      handle, construct_mapping(layout, std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc, 1, 2, 3, 2);

  // construct from all extents
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, std::extents<unsigned, 7>()), acc, 7);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, std::extents<size_t, D, 4, D>(2, 3)), acc, 2, 4, 3);
  test_mdspan_types<mec, ac>(handle, construct_mapping(layout, std::extents<char, D, 7, D>(0, 3)), acc, 0, 7, 3);
  test_mdspan_types<mec, ac>(
      handle, construct_mapping(layout, std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc, 1, 7, 2, 4, 3, 2);
}

template <bool ac, class H, class A>
constexpr void mixin_layout(const H& handle, const A& acc) {
  mixin_extents<true, ac>(handle, std::layout_left(), acc);
  mixin_extents<true, ac>(handle, std::layout_right(), acc);

  // Use weird layout, make sure it has the properties we want to test
  // Sanity check that this layouts mapping is constructible from extents (via its move constructor)
  static_assert(std::is_constructible_v<typename layout_wrapping_integral<8>::template mapping<std::extents<int>>,
                                        std::extents<int>>);
  static_assert(!std::is_constructible_v<typename layout_wrapping_integral<8>::template mapping<std::extents<int>>,
                                         const std::extents<int>&>);
  mixin_extents<true, ac>(handle, layout_wrapping_integral<8>(), acc);
  // Sanity check that this layouts mapping is not constructible from extents
  static_assert(!std::is_constructible_v<typename layout_wrapping_integral<4>::template mapping<std::extents<int>>,
                                         std::extents<int>>);
  static_assert(!std::is_constructible_v<typename layout_wrapping_integral<4>::template mapping<std::extents<int>>,
                                         const std::extents<int>&>);
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

  // test non-constructibility from wrong integer types
  constexpr size_t D = std::dynamic_extent;
  using mds_t        = std::mdspan<float, std::extents<int, 3, D, D>>;
  // sanity check
  static_assert(std::is_constructible_v<mds_t, float*, int, int, int>);
  static_assert(std::is_constructible_v<mds_t, float*, int, int>);
  // wrong number of arguments
  static_assert(!std::is_constructible_v<mds_t, float*, int>);
  static_assert(!std::is_constructible_v<mds_t, float*, int, int, int, int>);
  // not convertible to int
  static_assert(!std::is_constructible_v<mds_t, float*, int, int, std::dextents<int, 1>>);

  // test non-constructibility from wrong handle_type
  static_assert(!std::is_constructible_v<mds_t, const float*, int, int>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
