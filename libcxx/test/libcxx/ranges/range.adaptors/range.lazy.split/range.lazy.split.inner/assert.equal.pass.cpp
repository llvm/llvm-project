//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: availability-verbose_abort-missing

// <ranges>

// friend constexpr bool operator==(const inner-iterator& x, default_sentinel_t);
//
// Can't compare a default-constructed `inner-iterator` with the default sentinel.

#include <ranges>

#include "check_assertion.h"
#include "../types.h"

int main(int, char**) {
  {
    InnerIterForward i;
    TEST_LIBCPP_ASSERT_FAILURE(i == std::default_sentinel, "Cannot call comparison on a default-constructed iterator.");
  }

  {
    InnerIterInput i;
    TEST_LIBCPP_ASSERT_FAILURE(i == std::default_sentinel, "Cannot call comparison on a default-constructed iterator.");
  }

  return 0;
}
