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

// UNSUPPORTED: c++11, c++14

// UNSUPPORTED: c++03, windows, libcxx-no-debug-mode
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=0

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
