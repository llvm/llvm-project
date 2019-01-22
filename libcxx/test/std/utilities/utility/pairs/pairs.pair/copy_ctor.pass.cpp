//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// pair(const pair&) = default;

#include <utility>
#include <cassert>

#include "test_macros.h"

int main()
{
    {
        typedef std::pair<int, short> P1;
        P1 p1(3, static_cast<short>(4));
        P1 p2 = p1;
        assert(p2.first == 3);
        assert(p2.second == 4);
    }
#if TEST_STD_VER > 11
    {
        typedef std::pair<int, short> P1;
        constexpr P1 p1(3, static_cast<short>(4));
        constexpr P1 p2 = p1;
        static_assert(p2.first == 3, "");
        static_assert(p2.second == 4, "");
    }
#endif
}
