//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

// <algorithm>

// Calling `pop_heap` on an empty range is invalid.

#include <algorithm>

#include <array>
#include "check_assertion.h"

int main(int, char**) {
  std::array<int, 0> a;

  TEST_LIBCPP_ASSERT_FAILURE(std::pop_heap(a.begin(), a.end()), "The heap given to pop_heap must be non-empty");

  return 0;
}
