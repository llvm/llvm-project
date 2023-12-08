//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <optional>

// constexpr T* optional<T>::operator->();
// constexpr const T* optional<T>::operator->() const;

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: availability-verbose_abort-missing

#include <optional>

#include "check_assertion.h"

struct X {
    int test() const { return 3; }
};

int main(int, char**) {
    {
        std::optional<X> opt;
        TEST_LIBCPP_ASSERT_FAILURE(opt->test(), "optional operator-> called on a disengaged value");
    }

    {
        const std::optional<X> opt;
        TEST_LIBCPP_ASSERT_FAILURE(opt->test(), "optional operator-> called on a disengaged value");
    }

    return 0;
}
