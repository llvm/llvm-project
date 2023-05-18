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
// ADDITIONAL_COMPILE_FLAGS: -Wno-ctad-maybe-unsupported

// FIXME: https://github.com/llvm/llvm-project/issues/64719
// There appear to be some issues around ctad which make it
// currently impossible to get this code warning free.
// Thus added the additional compile flag above

// <mdspan>

// template<class OtherExtents>
//   constexpr explicit(extents_type::rank() > 0)
//     mapping(const layout_stride::mapping<OtherExtents>& other);
//
// Constraints: is_constructible_v<extents_type, OtherExtents> is true.
//
// Preconditions:
//   - If extents_type::rank() > 0 is true, then for all r in the range [0, extents_type::rank()),
//     other.stride(r) equals other.extents().fwd-prod-of-extents(r), and
//   - other.required_span_size() is representable as a value of type index_type ([basic.fundamental]).
//
// Effects: Direct-non-list-initializes extents_ with other.extents().

#include <mdspan>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
  constexpr size_t D = std::dynamic_extent;

  // working case
  {
    std::layout_stride::mapping arg(std::extents<int, D>(5), std::array<int, 1>{1});
    [[maybe_unused]] std::layout_right::mapping<std::extents<size_t, 5>> m(arg); // should work
  }
  // mismatch of static extent
  {
    std::layout_stride::mapping arg(std::extents<int, D>(5), std::array<int, 1>{1});
    TEST_LIBCPP_ASSERT_FAILURE(([=] { std::layout_right::mapping<std::extents<int, 3>> m(arg); }()),
                               "extents construction: mismatch of provided arguments with static extents.");
  }
  // non-representability of extents itself
  {
    std::layout_stride::mapping arg(std::extents<int, D>(500), std::array<int, 1>{1});
    TEST_LIBCPP_ASSERT_FAILURE(([=] { std::layout_right::mapping<std::extents<char, D>> m(arg); }()),
                               "extents ctor: arguments must be representable as index_type and nonnegative");
  }
  // non-representability of required span size
  {
    std::layout_stride::mapping arg(std::extents<int, D, D>(100, 3), std::array<int, 2>{3, 1});
    TEST_LIBCPP_ASSERT_FAILURE(
        ([=] { std::layout_right::mapping<std::extents<char, D, D>> m(arg); }()),
        "layout_right::mapping from layout_stride ctor: other.required_span_size() must be "
        "representable as index_type.");
  }
  // strides are not layout_right compatible
  {
    std::layout_stride::mapping arg(std::extents<int, D>(5), std::array<int, 1>{2});
    TEST_LIBCPP_ASSERT_FAILURE(
        ([=] { std::layout_right::mapping<std::extents<size_t, 5>> m(arg); }()),
        "layout_right::mapping from layout_stride ctor: strides are not compatible with layout_right.");
  }
  {
    std::layout_stride::mapping arg(std::extents<int, D, D>(100, 3), std::array<int, 2>{6, 2});
    TEST_LIBCPP_ASSERT_FAILURE(
        ([=] { std::layout_right::mapping<std::extents<int, D, D>> m(arg); }()),
        "layout_right::mapping from layout_stride ctor: strides are not compatible with layout_right.");
  }

  return 0;
}
