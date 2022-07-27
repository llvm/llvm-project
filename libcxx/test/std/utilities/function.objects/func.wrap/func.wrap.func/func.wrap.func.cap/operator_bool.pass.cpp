//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// class function<R(ArgTypes...)>

// explicit operator bool() const

// This test runs in C++03, but we have deprecated using std::function in C++03.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS -D_LIBCPP_ENABLE_CXX03_FUNCTION

#include <functional>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

int g(int) {return 0;}

int main(int, char**)
{
    static_assert(std::is_constructible<bool, std::function<void()> >::value, "");
    static_assert(!std::is_convertible<std::function<void()>, bool>::value, "");

    {
    std::function<int(int)> f;
    assert(!f);
    f = g;
    assert(f);
    }

  return 0;
}
