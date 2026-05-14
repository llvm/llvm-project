//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <array>

// Test that operator[] triggers an assertion when accessing the array out-of-bounds.

#include <array>

#include "check_assertion.h"

int main(int, char**) {
  // Check with an empty array
  {
    {
      using Array     = std::array<int, 0>;
      Array c         = {};
      Array const& cc = c;
      TEST_LIBCPP_ASSERT_FAILURE(c[0], "cannot call array<T, 0>::operator[] on a zero-sized array");
      TEST_LIBCPP_ASSERT_FAILURE(c[1], "cannot call array<T, 0>::operator[] on a zero-sized array");
      TEST_LIBCPP_ASSERT_FAILURE(cc[0], "cannot call array<T, 0>::operator[] on a zero-sized array");
      TEST_LIBCPP_ASSERT_FAILURE(cc[1], "cannot call array<T, 0>::operator[] on a zero-sized array");
    }
    {
      using Array     = std::array<const int, 0>;
      Array c         = {{}};
      Array const& cc = c;
      TEST_LIBCPP_ASSERT_FAILURE(c[0], "cannot call array<T, 0>::operator[] on a zero-sized array");
      TEST_LIBCPP_ASSERT_FAILURE(c[1], "cannot call array<T, 0>::operator[] on a zero-sized array");
      TEST_LIBCPP_ASSERT_FAILURE(cc[0], "cannot call array<T, 0>::operator[] on a zero-sized array");
      TEST_LIBCPP_ASSERT_FAILURE(cc[1], "cannot call array<T, 0>::operator[] on a zero-sized array");
    }
  }

  // Check with non-empty arrays
  {
    {
      using Array     = std::array<int, 1>;
      Array c         = {};
      Array const& cc = c;
      TEST_LIBCPP_ASSERT_FAILURE(c[2], "out-of-bounds access in std::array<T, N>");
      TEST_LIBCPP_ASSERT_FAILURE(cc[2], "out-of-bounds access in std::array<T, N>");
    }
    {
      using Array     = std::array<const int, 1>;
      Array c         = {{}};
      Array const& cc = c;
      TEST_LIBCPP_ASSERT_FAILURE(c[2], "out-of-bounds access in std::array<T, N>");
      TEST_LIBCPP_ASSERT_FAILURE(cc[2], "out-of-bounds access in std::array<T, N>");
    }

    {
      using Array     = std::array<int, 5>;
      Array c         = {};
      Array const& cc = c;
      TEST_LIBCPP_ASSERT_FAILURE(c[99], "out-of-bounds access in std::array<T, N>");
      TEST_LIBCPP_ASSERT_FAILURE(cc[99], "out-of-bounds access in std::array<T, N>");
    }
    {
      using Array     = std::array<const int, 5>;
      Array c         = {{}};
      Array const& cc = c;
      TEST_LIBCPP_ASSERT_FAILURE(c[99], "out-of-bounds access in std::array<T, N>");
      TEST_LIBCPP_ASSERT_FAILURE(cc[99], "out-of-bounds access in std::array<T, N>");
    }
  }

  return 0;
}
