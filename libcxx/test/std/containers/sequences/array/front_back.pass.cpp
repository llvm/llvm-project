//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <array>

// reference front();
// reference back();
// const_reference front(); // constexpr in C++14
// const_reference back(); // constexpr in C++14

#include <array>
#include <cassert>

#include "test_macros.h"

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

int main()
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        C c = {1, 2, 3.5};

        C::reference r1 = c.front();
        assert(r1 == 1);
        r1 = 5.5;
        assert(c[0] == 5.5);

        C::reference r2 = c.back();
        assert(r2 == 3.5);
        r2 = 7.5;
        assert(c[2] == 7.5);
    }
    {
        typedef double T;
        typedef std::array<T, 3> C;
        const C c = {1, 2, 3.5};
        C::const_reference r1 = c.front();
        assert(r1 == 1);

        C::const_reference r2 = c.back();
        assert(r2 == 3.5);
    }

#if TEST_STD_VER > 11
    {
        typedef double T;
        typedef std::array<T, 3> C;
        constexpr C c = {1, 2, 3.5};

        constexpr T t1 = c.front();
        static_assert (t1 == 1, "");

        constexpr T t2 = c.back();
        static_assert (t2 == 3.5, "");
    }
#endif

}
