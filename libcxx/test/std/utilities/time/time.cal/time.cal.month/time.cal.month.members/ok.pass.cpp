//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <chrono>
// class month;

// constexpr bool ok() const noexcept;
//  Returns: 1 <= d_ && d_ <= 12

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main()
{
    using month = std::chrono::month;

    ASSERT_NOEXCEPT(                std::declval<const month>().ok());
    ASSERT_SAME_TYPE(bool, decltype(std::declval<const month>().ok()));

    static_assert(!month{0}.ok(), "");
    static_assert( month{1}.ok(), "");

    assert(!month{0}.ok());
    for (unsigned i = 1; i <= 12; ++i)
        assert(month{i}.ok());
    for (unsigned i = 13; i <= 255; ++i)
        assert(!month{i}.ok());
}
