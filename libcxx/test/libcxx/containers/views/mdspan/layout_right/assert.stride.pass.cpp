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

// <mdspan>

// layout_right::mapping
//
// constexpr index_type stride(rank_type i) const noexcept;
//
//   Constraints: extents_type::rank() > 0 is true.
//
//   Preconditions: i < extents_type::rank() is true.
//
//   Returns: extents().rev-prod-of-extents(i).
//
//

#include <mdspan>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
  // value out of range
  {
    std::layout_right::mapping<std::dextents<int, 3>> m{std::dextents<int, 3>{100, 100, 100}};

    TEST_LIBCPP_ASSERT_FAILURE(m.stride(4), "layout_right::mapping::stride(): invalid rank index");
  }
  return 0;
}
