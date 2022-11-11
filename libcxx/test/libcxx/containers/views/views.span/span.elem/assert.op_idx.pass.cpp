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
// constexpr reference operator[](size_type idx) const;

// Make sure that accessing a span out-of-bounds triggers an assertion.

// REQUIRES: has-unix-headers
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

#include <array>
#include <span>

#include "check_assertion.h"

int main(int, char**) {
    {
        std::array<int, 3> array{0, 1, 2};
        std::span<int> const s(array.data(), array.size());
        TEST_LIBCPP_ASSERT_FAILURE(s[3], "span<T>::operator[](index): index out of range");
    }

    {
        std::array<int, 3> array{0, 1, 2};
        std::span<int, 3> const s(array.data(), array.size());
        TEST_LIBCPP_ASSERT_FAILURE(s[3], "span<T, N>::operator[](index): index out of range");
    }

    return 0;
}
