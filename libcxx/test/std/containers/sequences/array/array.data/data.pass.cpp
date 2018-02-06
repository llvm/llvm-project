//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <array>

// T *data();

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
        T* p = c.data();
        assert(p[0] == 1);
        assert(p[1] == 2);
        assert(p[2] == 3.5);
    }
    {
        typedef double T;
        typedef std::array<T, 0> C;
        C c = {};
        T* p = c.data();
        (void)p; // to placate scan-build
    }
}
