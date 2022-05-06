//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17, libcxx-no-debug-mode, libcpp-has-no-incomplete-ranges
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

// <ranges>

// friend constexpr bool operator==(const inner-iterator& x, default_sentinel_t);
//
// Can't compare a default-constructed `inner-iterator` with the default sentinel.

#include <ranges>

#include "check_assertion.h"
#include "../types.h"

int main(int, char**) {
  {
    OuterIterForward i;
    TEST_LIBCPP_ASSERT_FAILURE(i == std::default_sentinel, "Cannot call comparison on a default-constructed iterator.");
  }

  {
    OuterIterInput i;
    TEST_LIBCPP_ASSERT_FAILURE(i == std::default_sentinel, "Cannot call comparison on a default-constructed iterator.");
  }

  return 0;
}
