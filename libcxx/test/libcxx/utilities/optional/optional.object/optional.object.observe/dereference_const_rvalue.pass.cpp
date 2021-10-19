//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr T&& optional<T>::operator*() const &&;

// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <optional>

#include "test_macros.h"
#include "debug_macros.h"

int main(int, char**) {
    const std::optional<int> opt;
    TEST_LIBCPP_ASSERT_FAILURE(*std::move(opt), "optional operator* called on a disengaged value");

    return 0;
}
