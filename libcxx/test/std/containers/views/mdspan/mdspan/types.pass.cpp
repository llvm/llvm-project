//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>
//
//  template<class ElementType, class Extents, class LayoutPolicy = layout_right,
//           class AccessorPolicy = default_accessor<ElementType>>
//  class mdspan {
//  public:
//    using extents_type = Extents;
//    using layout_type = LayoutPolicy;
//    using accessor_type = AccessorPolicy;
//    using mapping_type = typename layout_type::template mapping<extents_type>;
//    using element_type = ElementType;
//    using value_type = remove_cv_t<element_type>;
//    using index_type = typename extents_type::index_type;
//    using size_type = typename extents_type::size_type;
//    using rank_type = typename extents_type::rank_type;
//    using data_handle_type = typename accessor_type::data_handle_type;
//    using reference = typename accessor_type::reference;
//    ...
//  };

#include <mdspan>
#include <cassert>
#include <concepts>
#include <span> // dynamic_extent
#include <type_traits>

#include "test_macros.h"

#include "../MinimalElementType.h"
#include "CustomTestAccessors.h"
#include "../CustomTestLayouts.h"

// Calculated expected size of an mdspan
// Note this expects that only default_accessor is empty
template<class MDS>
constexpr size_t expected_size() {
  size_t sizeof_dht = sizeof(typename MDS::data_handle_type);
  size_t result = sizeof_dht;
  if(MDS::rank_dynamic() > 0) {
    size_t alignof_idx = alignof(typename MDS::index_type);
    size_t sizeof_idx = sizeof(typename MDS::index_type);
    // add alignment if necessary
    result += sizeof_dht%alignof_idx == 0?0:alignof_idx - (sizeof_dht%alignof_idx);
    // add sizeof stored extents
    result += MDS::rank_dynamic() * sizeof_idx;
  }
  using A = typename MDS::accessor_type;
  if(!std::is_same_v<A, std::default_accessor<typename MDS::element_type>>) {
    size_t alignof_acc = alignof(A);
    size_t sizeof_acc = sizeof(A);
    // add alignment if necessary
    result += result%alignof_acc == 0?0:alignof_acc - (result%alignof_acc);
    // add sizeof stored accessor
    result += sizeof_acc;
  }
  // add alignment of the mdspan itself
  result += result%alignof(MDS) == 0?0:alignof(MDS) - (result%alignof(MDS));
  return result;
}

// check triviality
template <class T>
constexpr bool trv_df_ctor = std::is_trivially_default_constructible_v<T>;
template <class T>
constexpr bool trv_cp_ctor = std::is_trivially_copy_constructible_v<T>;
template <class T>
constexpr bool trv_mv_ctor = std::is_trivially_move_constructible_v<T>;
template <class T>
constexpr bool trv_dstruct = std::is_trivially_destructible_v<T>;
template <class T>
constexpr bool trv_cp_asgn = std::is_trivially_copy_assignable_v<T>;
template <class T>
constexpr bool trv_mv_asgn = std::is_trivially_move_assignable_v<T>;

template <class MDS, bool default_ctor, bool copy_ctor, bool move_ctor, bool destr, bool copy_assign, bool move_assign>
void check_triviality() {
  static_assert(trv_df_ctor<MDS> == default_ctor);
  static_assert(trv_cp_ctor<MDS> == copy_ctor);
  static_assert(trv_mv_ctor<MDS> == move_ctor);
  static_assert(trv_dstruct<MDS> == destr);
  static_assert(trv_cp_asgn<MDS> == copy_assign);
  static_assert(trv_mv_asgn<MDS> == move_assign);
}

template <class T, class E, class L, class A>
void test_mdspan_types() {
  using MDS = std::mdspan<T, E, L, A>;

  ASSERT_SAME_TYPE(typename MDS::extents_type, E);
  ASSERT_SAME_TYPE(typename MDS::layout_type, L);
  ASSERT_SAME_TYPE(typename MDS::accessor_type, A);
  ASSERT_SAME_TYPE(typename MDS::mapping_type, typename L::template mapping<E>);
  ASSERT_SAME_TYPE(typename MDS::element_type, T);
  ASSERT_SAME_TYPE(typename MDS::value_type, std::remove_cv_t<T>);
  ASSERT_SAME_TYPE(typename MDS::index_type, typename E::index_type);
  ASSERT_SAME_TYPE(typename MDS::size_type, typename E::size_type);
  ASSERT_SAME_TYPE(typename MDS::rank_type, typename E::rank_type);
  ASSERT_SAME_TYPE(typename MDS::data_handle_type, typename A::data_handle_type);
  ASSERT_SAME_TYPE(typename MDS::reference, typename A::reference);

// This miserably failed with clang-cl - likely because it doesn't honor/enable
// no-unique-address fully by default
#ifndef _WIN32
  // check the size of mdspan
  if constexpr (std::is_same_v<L, std::layout_left> || std::is_same_v<L, std::layout_right>) {
    LIBCPP_STATIC_ASSERT(sizeof(MDS) == expected_size<MDS>());
  }
#endif

  // check default template parameters:
  ASSERT_SAME_TYPE(std::mdspan<T, E>, std::mdspan<T, E, std::layout_right, std::default_accessor<T>>);
  ASSERT_SAME_TYPE(std::mdspan<T, E, L>, std::mdspan<T, E, L, std::default_accessor<T>>);

  // check triviality
  using DH = typename MDS::data_handle_type;
  using MP = typename MDS::mapping_type;

  check_triviality<MDS,
                   false, // mdspan is never trivially constructible right now
                   trv_cp_ctor<DH> && trv_cp_ctor<MP> && trv_cp_ctor<A>,
                   trv_mv_ctor<DH> && trv_mv_ctor<MP> && trv_mv_ctor<A>,
                   trv_dstruct<DH> && trv_dstruct<MP> && trv_dstruct<A>,
                   trv_cp_asgn<DH> && trv_cp_asgn<MP> && trv_cp_asgn<A>,
                   trv_mv_asgn<DH> && trv_mv_asgn<MP> && trv_mv_asgn<A>>();
}

template <class T, class L, class A>
void mixin_extents() {
  constexpr size_t D = std::dynamic_extent;
  test_mdspan_types<T, std::extents<int>, L, A>();
  test_mdspan_types<T, std::extents<signed char, D>, L, A>();
  test_mdspan_types<T, std::dextents<signed char, 7>, L, A>();
  test_mdspan_types<T, std::dextents<signed char, 9>, L, A>();
  test_mdspan_types<T, std::extents<unsigned, 7>, L, A>();
  test_mdspan_types<T, std::extents<unsigned, D, D, D>, L, A>();
  test_mdspan_types<T, std::extents<size_t, D, 7, D>, L, A>();
  test_mdspan_types<T, std::extents<int64_t, D, 7, D, 4, D, D>, L, A>();
}

template <class T, class A>
void mixin_layout() {
  mixin_extents<T, std::layout_left, A>();
  mixin_extents<T, std::layout_right, A>();
  mixin_extents<T, layout_wrapping_integral<4>, A>();
}

template <class T>
void mixin_accessor() {
  mixin_layout<T, std::default_accessor<T>>();
  mixin_layout<T, checked_accessor<T>>();
}

int main(int, char**) {
  mixin_accessor<int>();
  mixin_accessor<const int>();
  mixin_accessor<double>();
  mixin_accessor<const double>();
  mixin_accessor<MinimalElementType>();
  mixin_accessor<const MinimalElementType>();

  // sanity checks for triviality
  check_triviality<std::mdspan<int, std::extents<int>>, false, true, true, true, true, true>();
  check_triviality<std::mdspan<int, std::dextents<int, 1>>, false, true, true, true, true, true>();
  check_triviality<std::mdspan<int, std::dextents<int, 1>, std::layout_right, checked_accessor<int>>,
                   false,
                   true,
                   false,
                   true,
                   true,
                   true>();
  return 0;
}
