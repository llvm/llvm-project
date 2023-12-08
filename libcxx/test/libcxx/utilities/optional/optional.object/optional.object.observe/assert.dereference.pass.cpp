//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <optional>

// constexpr T& optional<T>::operator*() &;
// constexpr T&& optional<T>::operator*() &&;
// constexpr const T& optional<T>::operator*() const &;
// constexpr T&& optional<T>::operator*() const &&;

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: availability-verbose_abort-missing

#include <optional>

#include "check_assertion.h"

int main(int, char**) {
    // &
    {
        std::optional<int> opt;
        TEST_LIBCPP_ASSERT_FAILURE(*opt, "optional operator* called on a disengaged value");
    }

    // &&
    {
        std::optional<int> opt;
        TEST_LIBCPP_ASSERT_FAILURE(*std::move(opt), "optional operator* called on a disengaged value");
    }

    // const &
    {
        const std::optional<int> opt;
        TEST_LIBCPP_ASSERT_FAILURE(*opt, "optional operator* called on a disengaged value");
    }

    // const &&
    {
        const std::optional<int> opt;
        TEST_LIBCPP_ASSERT_FAILURE(*std::move(opt), "optional operator* called on a disengaged value");
    }

    return 0;
}
