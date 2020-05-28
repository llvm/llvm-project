//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// implicitly generated array constructors / assignment operators

#include <array>
#include <type_traits>
#include <cassert>
#include "test_macros.h"

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

// In C++03 the copy assignment operator is not deleted when the implicitly
// generated operator would be ill-formed; like in the case of a struct with a
// const member.
#if TEST_STD_VER < 11
#   define TEST_NOT_COPY_ASSIGNABLE(T) ((void)0)
#else
#   define TEST_NOT_COPY_ASSIGNABLE(T) static_assert(!std::is_copy_assignable<T>::value, "")
#endif

struct NoDefault {
    TEST_CONSTEXPR NoDefault(int) { }
};

TEST_CONSTEXPR_CXX14 bool tests()
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        C c = {1.1, 2.2, 3.3};
        C c2 = c;
        c2 = c;
        static_assert(std::is_copy_constructible<C>::value, "");
        static_assert(std::is_copy_assignable<C>::value, "");
    }
    {
        typedef double T;
        typedef std::array<const T, 3> C;
        C c = {1.1, 2.2, 3.3};
        C c2 = c;
        ((void)c2);
        static_assert(std::is_copy_constructible<C>::value, "");
        TEST_NOT_COPY_ASSIGNABLE(C);
    }
    {
        typedef double T;
        typedef std::array<T, 0> C;
        C c = {};
        C c2 = c;
        c2 = c;
        static_assert(std::is_copy_constructible<C>::value, "");
        static_assert(std::is_copy_assignable<C>::value, "");
    }
    {
        // const arrays of size 0 should disable the implicit copy assignment operator.
        typedef double T;
        typedef std::array<const T, 0> C;
        C c = {{}};
        C c2 = c;
        ((void)c2);
        static_assert(std::is_copy_constructible<C>::value, "");
        TEST_NOT_COPY_ASSIGNABLE(C);
    }
    {
        typedef NoDefault T;
        typedef std::array<T, 0> C;
        C c = {};
        C c2 = c;
        c2 = c;
        static_assert(std::is_copy_constructible<C>::value, "");
        static_assert(std::is_copy_assignable<C>::value, "");
    }
    {
        typedef NoDefault T;
        typedef std::array<const T, 0> C;
        C c = {{}};
        C c2 = c;
        ((void)c2);
        static_assert(std::is_copy_constructible<C>::value, "");
        TEST_NOT_COPY_ASSIGNABLE(C);
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
