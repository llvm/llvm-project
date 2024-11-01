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
// XFAIL: availability-verbose_abort-missing

// <mdspan>

// template<class OtherExtents>
//   constexpr explicit(!is_convertible_v<OtherExtents, extents_type>)
//     mapping(const layout_left::mapping<OtherExtents>&) noexcept;

// Constraints:
//   - extents_type::rank() <= 1 is true, and
//   - is_constructible_v<extents_type, OtherExtents> is true.
//
// Preconditions: other.required_span_size() is representable as a value of type index_type

#include <mdspan>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
  constexpr size_t D = std::dynamic_extent;
  std::extents<int, D> arg_exts{5};
  std::layout_left::mapping<std::extents<int, D>> arg(arg_exts);

  // working case
  {
    [[maybe_unused]] std::layout_right::mapping<std::extents<size_t, 5>> m(arg); // should work
  }
  // mismatch of static extent
  {
    TEST_LIBCPP_ASSERT_FAILURE(([=] { std::layout_right::mapping<std::extents<int, 3>> m(arg); }()),
                               "extents construction: mismatch of provided arguments with static extents.");
  }
  // non-representability of extents itself
  {
    TEST_LIBCPP_ASSERT_FAILURE(([=] { std::layout_right::mapping<std::extents<char, D>> m(
                                 std::layout_left::mapping<std::extents<int, D>>(std::extents<int, D>(500))); }()),
                               "extents ctor: arguments must be representable as index_type and nonnegative");
  }

  // Can't trigger required_span_size() representability assertion, since for rank-1 the above check will trigger first,
  // and this conversion constructor is constrained on rank <= 1.
  return 0;
}
