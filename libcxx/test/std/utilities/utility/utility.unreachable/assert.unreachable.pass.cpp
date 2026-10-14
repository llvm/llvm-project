//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-debug
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Make sure that reaching std::unreachable() with assertions enabled triggers an assertion.

#include <utility>

#include "check_assertion.h"

int main(int, char**) {
    TEST_LIBCPP_ASSERT_FAILURE(std::unreachable(), "std::unreachable() was reached");

    return 0;
}
