//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// template <class T, size_t N> constexpr size_type array<T,N>::size();

#include <array>
#include <cassert>

#include "test_macros.h"

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

int main(int, char**)
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        C c = {1, 2, 3.5};
        assert(c.size() == 3);
        assert(c.max_size() == 3);
        assert(!c.empty());
    }
    {
        typedef double T;
        typedef std::array<T, 0> C;
        C c = {};
        assert(c.size() == 0);
        assert(c.max_size() == 0);
        assert(c.empty());
    }
#if TEST_STD_VER >= 11
    {
        typedef double T;
        typedef std::array<T, 3> C;
        constexpr C c = {1, 2, 3.5};
        static_assert(c.size() == 3, "");
        static_assert(c.max_size() == 3, "");
        static_assert(!c.empty(), "");
    }
    {
        typedef double T;
        typedef std::array<T, 0> C;
        constexpr C c = {};
        static_assert(c.size() == 0, "");
        static_assert(c.max_size() == 0, "");
        static_assert(c.empty(), "");
    }
#endif

    return 0;
}
