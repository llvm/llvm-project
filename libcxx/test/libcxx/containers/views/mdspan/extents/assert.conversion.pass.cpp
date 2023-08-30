//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: !libcpp-has-hardened-mode && !libcpp-has-debug-mode && !libcpp-has-assertions
// XFAIL: availability-verbose_abort-missing

// <mdspan>

// template<class OtherIndexType, size_t... OtherExtents>
//     constexpr explicit(see below) extents(const extents<OtherIndexType, OtherExtents...>&) noexcept;
//
// Constraints:
//   * sizeof...(OtherExtents) == rank() is true.
//   * ((OtherExtents == dynamic_extent || Extents == dynamic_extent ||
//       OtherExtents == Extents) && ...) is true.
//
// Preconditions:
//   * other.extent(r) equals Er for each r for which Er is a static extent, and
//   * either
//      - sizeof...(OtherExtents) is zero, or
//      - other.extent(r) is representable as a value of type index_type for
//        every rank index r of other.
//
// Remarks: The expression inside explicit is equivalent to:
//          (((Extents != dynamic_extent) && (OtherExtents == dynamic_extent)) || ... ) ||
//          (numeric_limits<index_type>::max() < numeric_limits<OtherIndexType>::max())

#include <mdspan>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
  constexpr size_t D = std::dynamic_extent;
  std::extents<int, D, D> arg{1000, 5};

  // working case
  {
    [[maybe_unused]] std::extents<size_t, D, 5> e(arg); // should work
  }
  // mismatch of static extent
  {
    TEST_LIBCPP_ASSERT_FAILURE(([=] { std::extents<int, D, 3> e(arg); }()),
                               "extents construction: mismatch of provided arguments with static extents.");
  }
  // value out of range
  {
    TEST_LIBCPP_ASSERT_FAILURE(([=] { std::extents<char, D, 5> e(arg); }()),
                               "extents ctor: arguments must be representable as index_type and nonnegative");
  }
  return 0;
}
