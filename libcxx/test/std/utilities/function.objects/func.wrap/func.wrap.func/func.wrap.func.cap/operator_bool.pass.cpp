//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>

// class function<R(ArgTypes...)>

// explicit operator bool() const

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
