//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// void fill(const T& u);

#include <array>
#include <cassert>

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "test_macros.h"
#include "disable_missing_braces_warning.h"

TEST_CONSTEXPR_CXX20 bool tests()
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        C c = {1, 2, 3.5};
        c.fill(5.5);
        assert(c.size() == 3);
        assert(c[0] == 5.5);
        assert(c[1] == 5.5);
        assert(c[2] == 5.5);
    }

    {
        typedef double T;
        typedef std::array<T, 0> C;
        C c = {};
        c.fill(5.5);
        assert(c.size() == 0);
    }
    return true;
}

int main(int, char**)
{
    tests();
#if TEST_STD_VER >= 20
    static_assert(tests(), "");
#endif
    return 0;
}
