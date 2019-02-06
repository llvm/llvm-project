//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <chrono>
// class weekday_indexed;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main()
{
    using weekday_indexed = std::chrono::weekday_indexed;

    static_assert(std::is_trivially_copyable_v<weekday_indexed>, "");
    static_assert(std::is_standard_layout_v<weekday_indexed>, "");
}
