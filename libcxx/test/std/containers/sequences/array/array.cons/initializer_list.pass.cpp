//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// Construct with initizializer list

#include <array>
#include <cassert>

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

int main()
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        C c = {1, 2, 3.5};
        assert(c.size() == 3);
        assert(c[0] == 1);
        assert(c[1] == 2);
        assert(c[2] == 3.5);
    }
    {
        typedef double T;
        typedef std::array<T, 0> C;
        C c = {};
        assert(c.size() == 0);
    }

    {
        typedef double T;
        typedef std::array<T, 3> C;
        C c = {1};
        assert(c.size() == 3.0);
        assert(c[0] == 1);
    }
    {
        typedef int T;
        typedef std::array<T, 1> C;
        C c = {};
        assert(c.size() == 1);
    }
}
