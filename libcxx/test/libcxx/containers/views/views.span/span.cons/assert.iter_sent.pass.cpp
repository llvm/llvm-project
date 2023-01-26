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
// constexpr span<T, Extent>::span(Iterator it, Sentinel sent);
//
// Check that we ensure `Extent == sent - it` and also that `[it, sent)` is a valid range.
//
//
// constexpr span<T, dynamic_extent>::span(Iterator it, Sentinel sent);
//
// Check that we ensure that `[it, sent)` is a valid range.

// REQUIRES: has-unix-headers
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

#include <array>
#include <span>

#include "check_assertion.h"

int main(int, char**) {
    {
        std::array<int, 3> array{0, 1, 2};

        auto invalid_range = [&] { std::span<int> const s(array.end(), array.begin()); (void)s; };
        TEST_LIBCPP_ASSERT_FAILURE(invalid_range(), "invalid range in span's constructor (iterator, sentinel)");
    }
    {
        std::array<int, 3> array{0, 1, 2};

        auto invalid_range = [&] { std::span<int, 3> const s(array.end(), array.begin()); (void)s; };
        TEST_LIBCPP_ASSERT_FAILURE(invalid_range(), "invalid range in span's constructor (iterator, sentinel)");

        auto invalid_size = [&] { std::span<int, 3> const s(array.begin(), array.begin() + 2); (void)s; };
        TEST_LIBCPP_ASSERT_FAILURE(invalid_size(), "invalid range in span's constructor (iterator, sentinel): last - first != extent");
    }

    return 0;
}
