//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// const_reference operator[](size_type) const; // constexpr in C++14
// Libc++ marks it as noexcept

#include <array>
#include <cassert>

#include "test_macros.h"

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"


TEST_CONSTEXPR_CXX14 bool tests()
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        C const c = {1, 2, 3.5};
        LIBCPP_ASSERT_NOEXCEPT(c[0]);
        ASSERT_SAME_TYPE(C::const_reference, decltype(c[0]));
        C::const_reference r1 = c[0];
        assert(r1 == 1);
        C::const_reference r2 = c[2];
        assert(r2 == 3.5);
    }
    // Test operator[] "works" on zero sized arrays
    {
        {
            typedef double T;
            typedef std::array<T, 0> C;
            C const c = {};
            LIBCPP_ASSERT_NOEXCEPT(c[0]);
            ASSERT_SAME_TYPE(C::const_reference, decltype(c[0]));
            if (c.size() > (0)) { // always false
                C::const_reference r = c[0];
                (void)r;
            }
        }
        {
            typedef double T;
            typedef std::array<T const, 0> C;
            C const c = {};
            LIBCPP_ASSERT_NOEXCEPT(c[0]);
            ASSERT_SAME_TYPE(C::const_reference, decltype(c[0]));
            if (c.size() > (0)) { // always false
              C::const_reference r = c[0];
              (void)r;
            }
        }
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
