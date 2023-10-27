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

// static constexpr size_t static_extent(rank_type i) noexcept;
//
//   Preconditions: i < rank() is true.
//
//   Returns: Ei.
//
//
// constexpr index_type extent(rank_type i) const noexcept;
//
//   Preconditions: i < rank() is true.
//
//   Returns: Di.

#include <mdspan>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
  constexpr size_t D = std::dynamic_extent;

  // mismatch of static extent
  {
    std::extents<int> e;
    TEST_LIBCPP_ASSERT_FAILURE(([=] { e.extent(0); }()), "extents access: index must be less than rank");
    TEST_LIBCPP_ASSERT_FAILURE(([=] { e.static_extent(0); }()), "extents access: index must be less than rank");
  }
  {
    std::extents<int, D> e;
    TEST_LIBCPP_ASSERT_FAILURE(([=] { e.extent(2); }()), "extents access: index must be less than rank");
    TEST_LIBCPP_ASSERT_FAILURE(([=] { e.static_extent(2); }()), "extents access: index must be less than rank");
  }
  {
    std::extents<int, 5> e;
    TEST_LIBCPP_ASSERT_FAILURE(([=] { e.extent(2); }()), "extents access: index must be less than rank");
    TEST_LIBCPP_ASSERT_FAILURE(([=] { e.static_extent(2); }()), "extents access: index must be less than rank");
  }
  {
    std::extents<int, D, 5> e;
    TEST_LIBCPP_ASSERT_FAILURE(([=] { e.extent(2); }()), "extents access: index must be less than rank");
    TEST_LIBCPP_ASSERT_FAILURE(([=] { e.static_extent(2); }()), "extents access: index must be less than rank");
  }
  {
    std::extents<int, 1, 2, 3, 4, 5, 6, 7, 8> e;
    TEST_LIBCPP_ASSERT_FAILURE(([=] { e.extent(9); }()), "extents access: index must be less than rank");
    TEST_LIBCPP_ASSERT_FAILURE(([=] { e.static_extent(9); }()), "extents access: index must be less than rank");
  }

  // check that static_extent works in constant expression with assertions enabled
  static_assert(std::extents<int, D, 5>::static_extent(1) == 5);
  return 0;
}
