//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// array();

#include <array>
#include <cassert>

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "test_macros.h"
#include "disable_missing_braces_warning.h"

struct NoDefault {
    TEST_CONSTEXPR NoDefault(int) { }
};

struct Default {
    TEST_CONSTEXPR Default() { }
};

TEST_CONSTEXPR_CXX14 bool tests()
{
    {
        std::array<Default, 3> array;
        assert(array.size() == 3);
    }

    {
        std::array<Default, 0> array;
        assert(array.size() == 0);
    }

    {
        typedef std::array<NoDefault, 0> C;
        C c;
        assert(c.size() == 0);
        C c1 = {};
        assert(c1.size() == 0);
        C c2 = {{}};
        assert(c2.size() == 0);
    }

    return true;
}

int main(int, char**)
{
    tests();
#if TEST_STD_VER >= 14
    static_assert(tests(), "");
#endif
    return 0;
}
