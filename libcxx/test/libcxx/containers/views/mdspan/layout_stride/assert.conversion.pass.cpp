//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing
// XFAIL: libcpp-hardening-mode=debug && target=powerpc{{.*}}le-unknown-linux-gnu

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
#include <cassert>

#include "check_assertion.h"
#include "../../../../../std/containers/views/mdspan/CustomTestLayouts.h"

int main(int, char**) {
  constexpr size_t D = std::dynamic_extent;

  // working case
  {
    std::extents<int, D, D> arg_exts{100, 5};
    std::layout_stride::mapping<std::extents<int, D, D>> arg(arg_exts, std::array<int, 2>{1, 100});
    [[maybe_unused]] std::layout_stride::mapping<std::extents<size_t, D, 5>> m(arg); // should work
  }
  // mismatch of static extent
  {
    std::extents<int, D, D> arg_exts{100, 5};
    std::layout_stride::mapping<std::extents<int, D, D>> arg(arg_exts, std::array<int, 2>{1, 100});
    TEST_LIBCPP_ASSERT_FAILURE(([=] { std::layout_stride::mapping<std::extents<int, D, 3>> m(arg); }()),
                               "extents construction: mismatch of provided arguments with static extents.");
  }
  // non-representability of extents itself
  {
    TEST_LIBCPP_ASSERT_FAILURE(
        ([=] {
          std::layout_stride::mapping<std::extents<signed char, D>> m(
              std::layout_stride::mapping<std::extents<int, D>>(std::extents<int, D>(500), std::array<int, 1>{1}));
        }()),
        "extents ctor: arguments must be representable as index_type and nonnegative");
  }
  // all strides must be larger than zero
  {
    always_convertible_layout::mapping<std::dextents<int, 2>> offset_map(std::dextents<int, 2>{10, 10}, 100, -1);
    TEST_LIBCPP_ASSERT_FAILURE(([=] { std::layout_stride::mapping<std::extents<signed char, D, D>> m(offset_map); }()),
                               "layout_stride::mapping converting ctor: all strides must be greater than 0");
  }
  // required_span_size not representable, while individual extents are
  {
    std::extents<int, D, D> arg_exts{100, 5};
    std::layout_stride::mapping<std::extents<int, D, D>> arg(arg_exts, std::array<int, 2>{1, 100});
    // check extents would be constructible
    [[maybe_unused]] std::extents<signed char, D, 5> e(arg_exts);
    // but the product is not, so we can't use it for layout_stride
    TEST_LIBCPP_ASSERT_FAILURE(
        ([=] { std::layout_stride::mapping<std::extents<signed char, D, 5>> m(arg); }()),
        "layout_stride::mapping converting ctor: other.required_span_size() must be representable as index_type.");
  }
  // required_span_size not representable, while individual extents are, edge case
  {
    // required span size = (3-1)*50 + (10-1) * 3 + 1 = 128
    std::extents<int, D, D> arg_exts{3, 10};
    std::layout_stride::mapping<std::extents<int, D, D>> arg(arg_exts, std::array<int, 2>{50, 3});
    // sanity check:
    assert(arg.required_span_size() == 128);
    // check extents would be constructible
    [[maybe_unused]] std::extents<signed char, D, 10> e(arg_exts);
    // but the product is not, so we can't use it for layout_stride
    TEST_LIBCPP_ASSERT_FAILURE(
        ([=] { std::layout_stride::mapping<std::extents<signed char, D, 10>> m(arg); }()),
        "layout_stride::mapping converting ctor: other.required_span_size() must be representable as index_type.");
  }
  // base offset must be 0 (i.e. mapping(0,...,0)==0) for a strided layout with positive strides
  {
    always_convertible_layout::mapping<std::dextents<int, 2>> offset_map(std::dextents<int, 2>{10, 10}, 3);
    TEST_LIBCPP_ASSERT_FAILURE(([=] { std::layout_stride::mapping<std::extents<signed char, D, D>> m(offset_map); }()),
                               "layout_stride::mapping converting ctor: base offset of mapping must be zero.");
  }
  return 0;
}
