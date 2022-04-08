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

// UNSUPPORTED: c++11, c++14

// UNSUPPORTED: c++03, windows
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

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
