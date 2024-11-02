//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

// <algorithm>

// In a call to `ranges::clamp(val, low, high)`, `low` must be `<= high`.

#include <algorithm>
#include <functional>

#include "check_assertion.h"

int main(int, char**) {
  (void)std::ranges::clamp(1, 2, 0, std::ranges::greater{});
  TEST_LIBCPP_ASSERT_FAILURE(std::ranges::clamp(1, 2, 0), "Bad bounds passed to std::ranges::clamp");

  (void)std::ranges::clamp(1, 0, 2);
  TEST_LIBCPP_ASSERT_FAILURE(
      std::ranges::clamp(1, 0, 2, std::ranges::greater{}), "Bad bounds passed to std::ranges::clamp");

  (void)std::ranges::clamp(1, 1, 1); // Equal bounds should be fine.

  return 0;
}
