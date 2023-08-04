//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-hardening-mode=unchecked
// XFAIL: availability-verbose_abort-missing

// <mdspan>

// template<class OtherExtents>
//   constexpr explicit(!is_convertible_v<OtherExtents, extents_type>)
//     mapping(const mapping<OtherExtents>&) noexcept;

// Constraints: is_constructible_v<extents_type, OtherExtents> is true.
//
// Preconditions: other.required_span_size() is representable as a value of type index_type.

#include <mdspan>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
  constexpr size_t D = std::dynamic_extent;
  std::extents<int, D, D> arg_exts{100, 5};
  std::layout_right::mapping<std::extents<int, D, D>> arg(arg_exts);

  // working case
  {
    [[maybe_unused]] std::layout_right::mapping<std::extents<size_t, D, 5>> m(arg); // should work
  }
  // mismatch of static extent
  {
    TEST_LIBCPP_ASSERT_FAILURE(([=] { std::layout_right::mapping<std::extents<int, D, 3>> m(arg); }()),
                               "extents construction: mismatch of provided arguments with static extents.");
  }
  // non-representability of extents itself
  {
    TEST_LIBCPP_ASSERT_FAILURE(([=] { std::layout_right::mapping<std::extents<char, D>> m(
                                 std::layout_right::mapping<std::extents<int, D>>(std::extents<int, D>(500))); }()),
                               "extents ctor: arguments must be representable as index_type and nonnegative");
  }
  // required_span_size not representable, while individual extents are
  {
    // check extents would be constructible
    [[maybe_unused]] std::extents<char, D, 5> e(arg_exts);
    // but the product is not, so we can't use it for layout_right
    TEST_LIBCPP_ASSERT_FAILURE(
        ([=] { std::layout_right::template mapping<std::extents<char, D, 5>> m(arg); }()),
        "layout_right::mapping converting ctor: other.required_span_size() must be representable as index_type.");
  }
  return 0;
}
