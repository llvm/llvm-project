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
// constexpr span<element_type, dynamic_extent> first(size_type count) const;

// Make sure that creating a sub-span with an incorrect number of elements triggers an assertion.

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
        TEST_LIBCPP_ASSERT_FAILURE(s.first(4), "span<T>::first(count): count out of range");
        TEST_LIBCPP_ASSERT_FAILURE(s.first<4>(), "span<T>::first<Count>(): Count out of range");
    }
    {
        std::array<int, 3> array{0, 1, 2};
        std::span<int, 3> const s(array.data(), array.size());
        TEST_LIBCPP_ASSERT_FAILURE(s.first(4), "span<T, N>::first(count): count out of range");
        // s.first<4>() caught at compile-time (tested elsewhere)
    }

    return 0;
}
