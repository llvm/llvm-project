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
// constexpr reference front() const noexcept;

// Make sure that accessing a span out-of-bounds triggers an assertion.

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <array>
#include <span>

#include "check_assertion.h"

int main(int, char**) {
    {
        std::array<int, 3> array{0, 1, 2};
        std::span<int> const s(array.data(), 0);
        TEST_LIBCPP_ASSERT_FAILURE(s.front(), "span<T>::front() on empty span");
    }

    {
        std::array<int, 3> array{0, 1, 2};
        std::span<int, 0> const s(array.data(), 0);
        TEST_LIBCPP_ASSERT_FAILURE(s.front(), "span<T, N>::front() on empty span");
    }

    return 0;
}
