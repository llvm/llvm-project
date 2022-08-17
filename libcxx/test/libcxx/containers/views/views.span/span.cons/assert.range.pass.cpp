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
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

#include <span>
#include <vector>

#include "check_assertion.h"

int main(int, char**) {
    std::vector<int> vec{0, 1, 2}; // must use std::vector instead of std::array, because std::span has a special constructor from std::array

    auto invalid_size = [&] { std::span<int, 2> const s(vec); (void)s; };
    TEST_LIBCPP_ASSERT_FAILURE(invalid_size(), "size mismatch in span's constructor (range)");

    return 0;
}
