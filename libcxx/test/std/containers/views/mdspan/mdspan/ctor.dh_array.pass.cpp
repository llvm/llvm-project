//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template<class OtherIndexType, size_t N>
//   constexpr explicit(N != rank_dynamic())
//     mdspan(data_handle_type p, const array<OtherIndexType, N>& exts);
//
// Constraints:
//   - is_convertible_v<const OtherIndexType&, index_type> is true,
//   - (is_nothrow_constructible<index_type, const OtherIndexType&> && ...) is true,
//   - N == rank() || N == rank_dynamic() is true,
//   - is_constructible_v<mapping_type, extents_type> is true, and
//   - is_default_constructible_v<accessor_type> is true.
//
// Preconditions: [0, map_.required_span_size()) is an accessible range of p and acc_
//                for the values of map_ and acc_ after the invocation of this constructor.
//
// Effects:
//   - Direct-non-list-initializes ptr_ with std::move(p),
//   - direct-non-list-initializes map_ with extents_type(exts), and
//   - value-initializes acc_.

#include <array>
#include <concepts>
#include <cassert>
#include <mdspan>
#include <type_traits>

#include "test_macros.h"

#include "../ConvertibleToIntegral.h"
#include "../MinimalElementType.h"
#include "../CustomTestLayouts.h"
#include "CustomTestAccessors.h"

template <class Extents, size_t... Idxs>
constexpr auto array_from_extents(const Extents& exts, std::index_sequence<Idxs...>) {
  return std::array<typename Extents::index_type, Extents::rank()>{exts.extent(Idxs)...};
}

template <class MDS, class Exts>
concept check_mdspan_ctor_implicit = requires(MDS m, typename MDS::data_handle_type h, const Exts& exts) {
  m = {h, exts};
};

template <class H, class M, class A, size_t N>
constexpr void
test_mdspan_ctor_array(const H& handle, const M& map, const A&, std::array<typename M::index_type, N> exts) {
  using MDS = std::mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;
  if !consteval {
    move_counted_handle<typename MDS::element_type>::move_counter() = 0;
  }
  MDS m(handle, exts);
  if !consteval {
    if constexpr (std::is_same_v<H, move_counted_handle<typename MDS::element_type>>) {
      assert((H::move_counter() == 1));
    }
  }

  static_assert(!noexcept(MDS(handle, exts)));

  static_assert(check_mdspan_ctor_implicit<MDS, decltype(exts)> == (N == MDS::rank_dynamic()));

  assert(m.extents() == map.extents());
  if constexpr (std::equality_comparable<H>)
    assert(m.data_handle() == handle);
  if constexpr (std::equality_comparable<M>)
    assert(m.mapping() == map);
  if constexpr (std::equality_comparable<A>)
    assert(m.accessor() == A());
}

template <bool mec, bool ac, class H, class M, class A>
constexpr void test_mdspan_ctor(const H& handle, const M& map, const A& acc) {
  using MDS = std::mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;
  static_assert(mec == std::is_constructible_v<M, typename M::extents_type>);
  static_assert(ac == std::is_default_constructible_v<A>);
  if constexpr (mec && ac) {
    // test from all extents
    auto exts = array_from_extents(map.extents(), std::make_index_sequence<MDS::rank()>());
    test_mdspan_ctor_array(handle, map, acc, exts);

    // test from dynamic extents
    std::array<typename MDS::index_type, MDS::rank_dynamic()> exts_dynamic{};
    size_t r_dyn = 0;
    for (size_t r = 0; r < MDS::rank(); r++) {
      if (MDS::static_extent(r) == std::dynamic_extent)
        exts_dynamic[r_dyn++] = exts[r];
    }
    test_mdspan_ctor_array(handle, map, acc, exts_dynamic);
  } else {
    static_assert(!std::is_constructible_v<MDS, const H&, const std::array<typename MDS::index_type, MDS::rank()>&>);
  }
}

template <bool mec, bool ac, class H, class L, class A>
constexpr void mixin_extents(const H& handle, const L& layout, const A& acc) {
  constexpr size_t D = std::dynamic_extent;
  test_mdspan_ctor<mec, ac>(handle, construct_mapping(layout, std::extents<int>()), acc);
  test_mdspan_ctor<mec, ac>(handle, construct_mapping(layout, std::extents<char, D>(7)), acc);
  test_mdspan_ctor<mec, ac>(handle, construct_mapping(layout, std::extents<unsigned, 7>()), acc);
  test_mdspan_ctor<mec, ac>(handle, construct_mapping(layout, std::extents<size_t, D, 4, D>(2, 3)), acc);
  test_mdspan_ctor<mec, ac>(handle, construct_mapping(layout, std::extents<char, D, 7, D>(0, 3)), acc);
  test_mdspan_ctor<mec, ac>(
      handle, construct_mapping(layout, std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc);
}

template <bool ac, class H, class A>
constexpr void mixin_layout(const H& handle, const A& acc) {
  mixin_extents<true, ac>(handle, std::layout_left(), acc);
  mixin_extents<true, ac>(handle, std::layout_right(), acc);

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

  // test non-constructibility from wrong array type
  constexpr size_t D = std::dynamic_extent;
  using mds_t        = std::mdspan<float, std::extents<unsigned, 3, D, D>>;
  // sanity check
  static_assert(std::is_constructible_v<mds_t, float*, std::array<int, 3>>);
  static_assert(std::is_constructible_v<mds_t, float*, std::array<int, 2>>);
  // wrong size
  static_assert(!std::is_constructible_v<mds_t, float*, std::array<int, 1>>);
  static_assert(!std::is_constructible_v<mds_t, float*, std::array<int, 4>>);
  // not convertible to index_type
  static_assert(std::is_convertible_v<const IntType&, int>);
  static_assert(!std::is_convertible_v<const IntType&, unsigned>);
  static_assert(!std::is_constructible_v<mds_t, float*, std::array<IntType, 2>>);

  // index_type is not nothrow constructible
  using mds_uchar_t = std::mdspan<float, std::extents<unsigned char, 3, D, D>>;
  static_assert(std::is_convertible_v<IntType, unsigned char>);
  static_assert(std::is_convertible_v<const IntType&, unsigned char>);
  static_assert(!std::is_nothrow_constructible_v<unsigned char, const IntType&>);
  static_assert(!std::is_constructible_v<mds_uchar_t, float*, std::array<IntType, 2>>);

  // convertible from non-const to index_type but not  from const
  using mds_int_t = std::mdspan<float, std::extents<int, 3, D, D>>;
  static_assert(std::is_convertible_v<IntTypeNC, int>);
  static_assert(!std::is_convertible_v<const IntTypeNC&, int>);
  static_assert(std::is_nothrow_constructible_v<int, IntTypeNC>);
  static_assert(!std::is_constructible_v<mds_int_t, float*, std::array<IntTypeNC, 2>>);

  // can't test a combo where std::is_nothrow_constructible_v<int, const IntTypeNC&> is true,
  // but std::is_convertible_v<const IntType&, int> is false

  // test non-constructibility from wrong handle_type
  static_assert(!std::is_constructible_v<mds_t, const float*, std::array<int, 2>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
