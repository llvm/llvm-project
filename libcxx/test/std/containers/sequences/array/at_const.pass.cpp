//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// const_reference at (size_type) const; // constexpr in C++14

#include <array>
#include <cassert>

#ifndef TEST_HAS_NO_EXCEPTIONS
#include <stdexcept>
#endif

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
        typename C::const_reference r1 = c.at(0);
        assert(r1 == 1);

        typename C::const_reference r2 = c.at(2);
        assert(r2 == 3.5);
    }
    return true;
}

void test_exceptions()
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        std::array<int, 4> const array = {1, 2, 3, 4};

        try {
            TEST_IGNORE_NODISCARD array.at(4);
            assert(false);
        } catch (std::out_of_range const&) {
            // pass
        } catch (...) {
            assert(false);
        }

        try {
            TEST_IGNORE_NODISCARD array.at(5);
            assert(false);
        } catch (std::out_of_range const&) {
            // pass
        } catch (...) {
            assert(false);
        }

        try {
            TEST_IGNORE_NODISCARD array.at(6);
            assert(false);
        } catch (std::out_of_range const&) {
            // pass
        } catch (...) {
            assert(false);
        }

        try {
            using size_type = decltype(array)::size_type;
            TEST_IGNORE_NODISCARD array.at(static_cast<size_type>(-1));
            assert(false);
        } catch (std::out_of_range const&) {
            // pass
        } catch (...) {
            assert(false);
        }
    }

    {
        std::array<int, 0> array = {};

        try {
            TEST_IGNORE_NODISCARD array.at(0);
            assert(false);
        } catch (std::out_of_range const&) {
            // pass
        } catch (...) {
            assert(false);
        }
    }
#endif
}

int main(int, char**)
{
    tests();
    test_exceptions();

#if TEST_STD_VER >= 14
    static_assert(tests(), "");
#endif
    return 0;
}
