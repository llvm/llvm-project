//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>
//
// constexpr span<T, Extent>::span(Range&& r);
//
// Check that we ensure `size(r) == Extent`.

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-hardened-mode && !libcpp-has-debug-mode && !libcpp-has-assertions
// XFAIL: availability-verbose_abort-missing

#include <span>
#include <vector>

#include "check_assertion.h"

int main(int, char**) {
    std::vector<int> vec{0, 1, 2}; // must use std::vector instead of std::array, because std::span has a special constructor from std::array

    auto invalid_size = [&] { std::span<int, 2> const s(vec); (void)s; };
    TEST_LIBCPP_ASSERT_FAILURE(invalid_size(), "size mismatch in span's constructor (range)");

    return 0;
}
