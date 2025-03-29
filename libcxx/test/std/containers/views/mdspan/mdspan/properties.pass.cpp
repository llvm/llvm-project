//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template<class ElementType, class Extents, class LayoutPolicy = layout_right,
//            class AccessorPolicy = default_accessor<ElementType>>
//   class mdspan {
//   public:
//     static constexpr rank_type rank() noexcept { return extents_type::rank(); }
//     static constexpr rank_type rank_dynamic() noexcept { return extents_type::rank_dynamic(); }
//     static constexpr size_t static_extent(rank_type r) noexcept
//       { return extents_type::static_extent(r); }
//     constexpr index_type extent(rank_type r) const noexcept { return extents().extent(r); }
//
//     constexpr size_type size() const noexcept;
//     [[nodiscard]] constexpr bool empty() const noexcept;
//
//
//     constexpr const extents_type& extents() const noexcept { return map_.extents(); }
//     constexpr const data_handle_type& data_handle() const noexcept { return ptr_; }
//     constexpr const mapping_type& mapping() const noexcept { return map_; }
//     constexpr const accessor_type& accessor() const noexcept { return acc_; }
//     /* per LWG-4021 "mdspan::is_always_meow() should be noexcept" */
//     static constexpr bool is_always_unique() noexcept
//       { return mapping_type::is_always_unique(); }
//     static constexpr bool is_always_exhaustive() noexcept
//       { return mapping_type::is_always_exhaustive(); }
//     static constexpr bool is_always_strided() noexcept
//       { return mapping_type::is_always_strided(); }
//
//     constexpr bool is_unique() const
//       { return map_.is_unique(); }
//     constexpr bool is_exhaustive() const
//       { return map_.is_exhaustive(); }
//     constexpr bool is_strided() const
//       { return map_.is_strided(); }
//     constexpr index_type stride(rank_type r) const
//       { return map_.stride(r); }
//   };
//
// Each specialization MDS of mdspan models copyable and
//    - is_nothrow_move_constructible_v<MDS> is true,
//    - is_nothrow_move_assignable_v<MDS> is true, and
//    - is_nothrow_swappable_v<MDS> is true.
// A specialization of mdspan is a trivially copyable type if its accessor_type, mapping_type, and data_handle_type are trivially copyable types.

#include <mdspan>
#include <cassert>
#include <concepts>
#include <span> // dynamic_extent
#include <type_traits>

#include "test_macros.h"

#include "../MinimalElementType.h"
#include "../CustomTestLayouts.h"

template <class H, class M, class A>
constexpr void test_mdspan_types(const H& handle, const M& map, const A& acc) {
  using MDS = std::mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;
  MDS m(handle, map, acc);

  // =====================================
  // Traits for every mdspan
  // =====================================
  static_assert(std::copyable<MDS>);
  static_assert(std::is_nothrow_move_constructible_v<MDS>);
  static_assert(std::is_nothrow_move_assignable_v<MDS>);
  static_assert(std::is_nothrow_swappable_v<MDS>);

  // =====================================
  // Invariants coming from data handle
  // =====================================
  // data_handle()
  ASSERT_SAME_TYPE(decltype(m.data_handle()), const H&);
  ASSERT_NOEXCEPT(m.data_handle());
  if constexpr (std::equality_comparable<H>) {
    assert(m.data_handle() == handle);
  }

  // =====================================
  // Invariants coming from extents
  // =====================================

  // extents()
  ASSERT_SAME_TYPE(decltype(m.extents()), const typename MDS::extents_type&);
  ASSERT_NOEXCEPT(m.extents());
  assert(m.extents() == map.extents());

  // rank()
  ASSERT_SAME_TYPE(decltype(m.rank()), typename MDS::rank_type);
  ASSERT_NOEXCEPT(m.rank());
  assert(MDS::rank() == MDS::extents_type::rank());

  // rank_dynamic()
  ASSERT_SAME_TYPE(decltype(m.rank_dynamic()), typename MDS::rank_type);
  ASSERT_NOEXCEPT(m.rank_dynamic());
  assert(MDS::rank_dynamic() == MDS::extents_type::rank_dynamic());

  // extent(r), static_extent(r), size()
  if constexpr (MDS::rank() > 0) {
    typename MDS::size_type size = 1;
    for (typename MDS::rank_type r = 0; r < MDS::rank(); r++) {
      ASSERT_SAME_TYPE(decltype(MDS::static_extent(r)), size_t);
      ASSERT_NOEXCEPT(MDS::static_extent(r));
      assert(MDS::static_extent(r) == MDS::extents_type::static_extent(r));
      ASSERT_SAME_TYPE(decltype(m.extent(r)), typename MDS::index_type);
      ASSERT_NOEXCEPT(m.extent(r));
      assert(m.extent(r) == m.extents().extent(r));
      size *= m.extent(r);
    }
    assert(m.size() == size);
  } else {
    assert(m.size() == 1);
  }
  ASSERT_SAME_TYPE(decltype(m.size()), typename MDS::size_type);
  ASSERT_NOEXCEPT(m.size());

  // empty()
  ASSERT_SAME_TYPE(decltype(m.empty()), bool);
  ASSERT_NOEXCEPT(m.empty());
  assert(m.empty() == (m.size() == 0));

  // =====================================
  // Invariants coming from mapping
  // =====================================

  // mapping()
  ASSERT_SAME_TYPE(decltype(m.mapping()), const M&);
  ASSERT_NOEXCEPT(m.mapping());

  // is_[always_]unique/exhaustive/strided()
  ASSERT_SAME_TYPE(decltype(MDS::is_always_unique()), bool);
  ASSERT_SAME_TYPE(decltype(MDS::is_always_exhaustive()), bool);
  ASSERT_SAME_TYPE(decltype(MDS::is_always_strided()), bool);
  ASSERT_SAME_TYPE(decltype(m.is_unique()), bool);
  ASSERT_SAME_TYPE(decltype(m.is_exhaustive()), bool);
  ASSERT_SAME_TYPE(decltype(m.is_strided()), bool);
  // per LWG-4021 "mdspan::is_always_meow() should be noexcept"
  static_assert(noexcept(MDS::is_always_unique()));
  static_assert(noexcept(MDS::is_always_exhaustive()));
  static_assert(noexcept(MDS::is_always_strided()));
  LIBCPP_STATIC_ASSERT(!noexcept(m.is_unique()));
  LIBCPP_STATIC_ASSERT(!noexcept(m.is_exhaustive()));
  LIBCPP_STATIC_ASSERT(!noexcept(m.is_strided()));
  static_assert(MDS::is_always_unique() == M::is_always_unique());
  static_assert(MDS::is_always_exhaustive() == M::is_always_exhaustive());
  static_assert(MDS::is_always_strided() == M::is_always_strided());
  assert(m.is_unique() == map.is_unique());
  assert(m.is_exhaustive() == map.is_exhaustive());
  assert(m.is_strided() == map.is_strided());

  // stride(r)
  if constexpr (MDS::rank() > 0) {
    if (m.is_strided()) {
      for (typename MDS::rank_type r = 0; r < MDS::rank(); r++) {
        ASSERT_SAME_TYPE(decltype(m.stride(r)), typename MDS::index_type);
        LIBCPP_STATIC_ASSERT(!noexcept(m.stride(r)));
        assert(m.stride(r) == map.stride(r));
      }
    }
  }

  // =====================================
  // Invariants coming from accessor
  // =====================================

  // accessor()
  ASSERT_SAME_TYPE(decltype(m.accessor()), const A&);
  ASSERT_NOEXCEPT(m.accessor());
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
