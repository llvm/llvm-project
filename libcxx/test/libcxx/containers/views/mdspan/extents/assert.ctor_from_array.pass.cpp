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

// <mdspan>

// Test construction from array:
//
// template<class OtherIndexType, size_t N>
//     constexpr explicit(N != rank_dynamic()) extents(const array<OtherIndexType, N>& exts) noexcept;
//
// Constraints:
//   * is_convertible_v<const OtherIndexType&, index_type> is true,
//   * is_nothrow_constructible_v<index_type, const OtherIndexType&> is true, and
//   * N == rank_dynamic() || N == rank() is true.
//
// Preconditions:
//   * If N != rank_dynamic() is true, exts[r] equals Er for each r for which
//     Er is a static extent, and
//   * either
//     - N is zero, or
//     - exts[r] is nonnegative and is representable as a value of type index_type
//       for every rank index r.
//

#include <mdspan>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
  constexpr size_t D = std::dynamic_extent;
  // working case
  {
    [[maybe_unused]] std::extents<int, D, 5> e1(std::array{1000, 5}); // should work
  }
  // mismatch of static extent
  {
    TEST_LIBCPP_ASSERT_FAILURE(
        ([] {
          std::extents<int, D, 5> e1(std::array{1000, 3});
        }()),
        "extents construction: mismatch of provided arguments with static extents.");
  }
  // value out of range
  {
    TEST_LIBCPP_ASSERT_FAILURE(([] { std::extents<signed char, D, 5> e1(std::array{1000, 5}); }()),
                               "extents ctor: arguments must be representable as index_type and nonnegative");
  }
  // negative value
  {
    TEST_LIBCPP_ASSERT_FAILURE(([] { std::extents<signed char, D, 5> e1(std::array{-1, 5}); }()),
                               "extents ctor: arguments must be representable as index_type and nonnegative");
  }
  return 0;
}
