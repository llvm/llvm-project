//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <optional>

// Add to iterator out of bounds.

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-hardening-mode=none, libcpp-has-abi-bounded-iterators-in-optional

#include <optional>

#include "check_assertion.h"

int main(int, char**) {
  {
    std::optional<int> opt(1);
    auto i = opt.begin();

    TEST_LIBCPP_ASSERT_FAILURE(
        i += 2,
        "__capacity_aware_iterator::operator+=: Attempting to advance iterator past its container's possible range");

    TEST_LIBCPP_ASSERT_FAILURE(
        i += -2, "__capacity_aware_iterator::operator+=: Attempting to rewind iterator past its container's start");

    TEST_LIBCPP_ASSERT_FAILURE(
        i -= 2, "__capacity_aware_iterator::operator-=: Attempting to rewind iterator before its container's start");

    TEST_LIBCPP_ASSERT_FAILURE(
        i -= -2,
        "__capacity_aware_iterator::operator+=: Attempting to advance iterator past its container's possible range");

    TEST_LIBCPP_ASSERT_FAILURE(
        i[2],
        "__capacity_aware_iterator::operator[]: Attempting to index iterator past its container's possible range");

    TEST_LIBCPP_ASSERT_FAILURE(
        i[-2], "__capacity_aware_iterator::operator[]: Attempting to index iterator before its container's start");
  }
}
