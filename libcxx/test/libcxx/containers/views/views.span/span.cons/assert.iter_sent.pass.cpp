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
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: availability-verbose_abort-missing

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
