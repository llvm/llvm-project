//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// constexpr unspecified ignore;

// UNSUPPORTED: c++03

#include <cassert>
#include <tuple>
#include <type_traits>

#include "test_macros.h"

constexpr bool test_ignore_constexpr()
{
#if TEST_STD_VER > 11
    { // Test that std::ignore provides constexpr converting assignment.
        auto& res = (std::ignore = 42);
        assert(&res == &std::ignore);
    }
    { // Test that std::ignore provides constexpr copy/move constructors
        auto copy = std::ignore;
        auto moved = std::move(copy);
        ((void)moved);
    }
    { // Test that std::ignore provides constexpr copy/move assignment
        auto copy = std::ignore;
        copy = std::ignore;
        auto moved = std::ignore;
        moved = std::move(copy);
    }
#endif
    return true;
}

int main(int, char**) {
    {
        constexpr auto& ignore_v = std::ignore;
        ((void)ignore_v);
    }
    {
        static_assert(test_ignore_constexpr(), "");
    }
    {
        LIBCPP_STATIC_ASSERT(std::is_trivial<decltype(std::ignore)>::value, "");
    }

  return 0;
}
