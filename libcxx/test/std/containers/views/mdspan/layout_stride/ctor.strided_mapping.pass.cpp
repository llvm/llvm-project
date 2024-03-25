//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template<class StridedLayoutMapping>
//   constexpr explicit(see below)
//     mapping(const StridedLayoutMapping& other) noexcept;
//
// Constraints:
//   - layout-mapping-alike<StridedLayoutMapping> is satisfied.
//   - is_constructible_v<extents_type, typename StridedLayoutMapping::extents_type> is true.
//   - StridedLayoutMapping::is_always_unique() is true.
//   - StridedLayoutMapping::is_always_strided() is true.
//
// Preconditions:
//   - StridedLayoutMapping meets the layout mapping requirements ([mdspan.layout.policy.reqmts]),
//   - other.stride(r) > 0 is true for every rank index r of extents(),
//   - other.required_span_size() is representable as a value of type index_type ([basic.fundamental]), and
//   - OFFSET(other) == 0 is true.
//
// Effects: Direct-non-list-initializes extents_ with other.extents(), and for all d in the range [0, rank_),
//          direct-non-list-initializes strides_[d] with other.stride(d).
//
// Remarks: The expression inside explicit is equivalent to:
//   - !(is_convertible_v<typename StridedLayoutMapping::extents_type, extents_type> &&
//       (is-mapping-of<layout_left, LayoutStrideMapping> ||
//        is-mapping-of<layout_right, LayoutStrideMapping> ||
//        is-mapping-of<layout_stride, LayoutStrideMapping>))

#include <mdspan>
#include <type_traits>
#include <cassert>
#include <limits>

#include "test_macros.h"

#include "../CustomTestLayouts.h"

template <bool implicit, class FromL, class ToE, class FromE>
constexpr void test_conversion(FromE src_exts) {
  using To   = std::layout_stride::mapping<ToE>;
  using From = typename FromL::template mapping<FromE>;

  From src([&]() {
    if constexpr (std::is_same_v<FromL, std::layout_stride>) {
      // just construct some strides which aren't layout_left/layout_right
      std::array<size_t, FromE::rank()> strides;
      size_t stride = 2;
      for (size_t r = 0; r < FromE::rank(); r++) {
        strides[r] = stride;
        stride *= src_exts.extent(r);
      }
      return From(src_exts, strides);
    } else {
      return From(src_exts);
    }
  }());

  ASSERT_NOEXCEPT(To(src));
  To dest(src);
  assert(dest == src);

  if constexpr (implicit) {
    To dest_implicit = src;
    assert(dest_implicit == src);
  } else {
    assert((!std::is_convertible_v<From, To>));
  }
}

template <class FromL, class T1, class T2>
constexpr void test_conversion() {
  constexpr size_t D = std::dynamic_extent;
  constexpr bool idx_convertible =
      static_cast<size_t>(std::numeric_limits<T1>::max()) >= static_cast<size_t>(std::numeric_limits<T2>::max());
  constexpr bool l_convertible =
      std::is_same_v<FromL, std::layout_right> || std::is_same_v<FromL, std::layout_left> ||
      std::is_same_v<FromL, std::layout_stride>;
  constexpr bool idx_l_convertible = idx_convertible && l_convertible;

  // clang-format off
  // adding extents convertibility expectation
  test_conversion<idx_l_convertible && true,  FromL, std::extents<T1>>(std::extents<T2>());
  test_conversion<idx_l_convertible && true,  FromL, std::extents<T1, D>>(std::extents<T2, D>(0));
  test_conversion<idx_l_convertible && true,  FromL, std::extents<T1, D>>(std::extents<T2, D>(5));
  test_conversion<idx_l_convertible && false, FromL, std::extents<T1, 5>>(std::extents<T2, D>(5));
  test_conversion<idx_l_convertible && true,  FromL, std::extents<T1, 5>>(std::extents<T2, 5>());
  test_conversion<idx_l_convertible && false, FromL, std::extents<T1, 5, D>>(std::extents<T2, D, D>(5, 5));
  test_conversion<idx_l_convertible && true,  FromL, std::extents<T1, D, D>>(std::extents<T2, D, D>(5, 5));
  test_conversion<idx_l_convertible && true,  FromL, std::extents<T1, D, D>>(std::extents<T2, D, 7>(5));
  test_conversion<idx_l_convertible && true,  FromL, std::extents<T1, 5, 7>>(std::extents<T2, 5, 7>());
  test_conversion<idx_l_convertible && false, FromL, std::extents<T1, 5, D, 8, D, D>>(std::extents<T2, D, D, 8, 9, 1>(5, 7));
  test_conversion<idx_l_convertible && true,  FromL, std::extents<T1, D, D, D, D, D>>(
                                                     std::extents<T2, D, D, D, D, D>(5, 7, 8, 9, 1));
  test_conversion<idx_l_convertible && true,  FromL, std::extents<T1, D, D, 8, 9, D>>(std::extents<T2, D, 7, 8, 9, 1>(5));
  test_conversion<idx_l_convertible && true,  FromL, std::extents<T1, 5, 7, 8, 9, 1>>(std::extents<T2, 5, 7, 8, 9, 1>());
  // clang-format on
}

template <class IdxT, size_t... Extents>
using ToM = std::layout_stride::mapping<std::extents<IdxT, Extents...>>;

template <class FromL, class IdxT, size_t... Extents>
using FromM = typename FromL::template mapping<std::extents<IdxT, Extents...>>;

template <class FromL>
constexpr void test_no_implicit_conversion() {
  constexpr size_t D = std::dynamic_extent;

  // Sanity check that one static to dynamic conversion works
  static_assert(std::is_constructible_v<ToM<int, D>, FromM<FromL, int, 5>>);
  static_assert(std::is_convertible_v<FromM<FromL, int, 5>, ToM<int, D>>);

  // Check that dynamic to static conversion only works explicitly
  static_assert(std::is_constructible_v<ToM<int, 5>, FromM<FromL, int, D>>);
  static_assert(!std::is_convertible_v<FromM<FromL, int, D>, ToM<int, 5>>);

  // Sanity check that one static to dynamic conversion works
  static_assert(std::is_constructible_v<ToM<int, D, 7>, FromM<FromL, int, 5, 7>>);
  static_assert(std::is_convertible_v<FromM<FromL, int, 5, 7>, ToM<int, D, 7>>);

  // Check that dynamic to static conversion only works explicitly
  static_assert(std::is_constructible_v<ToM<int, 5, 7>, FromM<FromL, int, D, 7>>);
  static_assert(!std::is_convertible_v<FromM<FromL, int, D, 7>, ToM<int, 5, 7>>);

  // Sanity check that smaller index_type to larger index_type conversion works
  static_assert(std::is_constructible_v<ToM<size_t, 5>, FromM<FromL, int, 5>>);
  static_assert(std::is_convertible_v<FromM<FromL, int, 5>, ToM<size_t, 5>>);

  // Check that larger index_type to smaller index_type conversion works explicitly only
  static_assert(std::is_constructible_v<ToM<int, 5>, FromM<FromL, size_t, 5>>);
  static_assert(!std::is_convertible_v<FromM<FromL, size_t, 5>, ToM<int, 5>>);
}

template <class FromL>
constexpr void test_rank_mismatch() {
  constexpr size_t D = std::dynamic_extent;

  static_assert(!std::is_constructible_v<ToM<int, D>, FromM<FromL, int>>);
  static_assert(!std::is_constructible_v<ToM<int>, FromM<FromL, int, D, D>>);
  static_assert(!std::is_constructible_v<ToM<int, D>, FromM<FromL, int, D, D>>);
  static_assert(!std::is_constructible_v<ToM<int, D, D, D>, FromM<FromL, int, D, D>>);
}

template <class FromL>
constexpr void test_static_extent_mismatch() {
  constexpr size_t D = std::dynamic_extent;

  static_assert(!std::is_constructible_v<ToM<int, D, 5>, FromM<FromL, int, D, 4>>);
  static_assert(!std::is_constructible_v<ToM<int, 5>, FromM<FromL, int, 4>>);
  static_assert(!std::is_constructible_v<ToM<int, 5, D>, FromM<FromL, int, 4, D>>);
}

template <class FromL>
constexpr void test_layout() {
  test_conversion<FromL, int, int>();
  test_conversion<FromL, int, size_t>();
  test_conversion<FromL, size_t, int>();
  test_conversion<FromL, size_t, long>();
  // the implicit convertibility test doesn't apply to non std::layouts
  if constexpr (!std::is_same_v<FromL, always_convertible_layout>)
    test_no_implicit_conversion<FromL>();
  test_rank_mismatch<FromL>();
  test_static_extent_mismatch<FromL>();
}

constexpr bool test() {
  test_layout<std::layout_right>();
  test_layout<std::layout_left>();
  test_layout<std::layout_stride>();
  test_layout<always_convertible_layout>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
