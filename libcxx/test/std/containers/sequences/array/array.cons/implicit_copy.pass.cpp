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

struct NonTrivialCopy {
    TEST_CONSTEXPR NonTrivialCopy() { }
    TEST_CONSTEXPR NonTrivialCopy(NonTrivialCopy const&) { }
    TEST_CONSTEXPR_CXX14 NonTrivialCopy& operator=(NonTrivialCopy const&) { return *this; }
};

TEST_CONSTEXPR_CXX14 bool tests()
{
    {
        typedef std::array<double, 3> Array;
        Array array = {1.1, 2.2, 3.3};
        Array copy = array;
        copy = array;
        static_assert(std::is_copy_constructible<Array>::value, "");
        static_assert(std::is_copy_assignable<Array>::value, "");
    }
    {
        typedef std::array<double const, 3> Array;
        Array array = {1.1, 2.2, 3.3};
        Array copy = array; (void)copy;
        static_assert(std::is_copy_constructible<Array>::value, "");
        TEST_NOT_COPY_ASSIGNABLE(Array);
    }
    {
        typedef std::array<double, 0> Array;
        Array array = {};
        Array copy = array;
        copy = array;
        static_assert(std::is_copy_constructible<Array>::value, "");
        static_assert(std::is_copy_assignable<Array>::value, "");
    }
    {
        // const arrays of size 0 should disable the implicit copy assignment operator.
        typedef std::array<double const, 0> Array;
        Array array = {};
        Array copy = array; (void)copy;
        static_assert(std::is_copy_constructible<Array>::value, "");
        TEST_NOT_COPY_ASSIGNABLE(Array);
    }
    {
        typedef std::array<NoDefault, 0> Array;
        Array array = {};
        Array copy = array;
        copy = array;
        static_assert(std::is_copy_constructible<Array>::value, "");
        static_assert(std::is_copy_assignable<Array>::value, "");
    }
    {
        typedef std::array<NoDefault const, 0> Array;
        Array array = {};
        Array copy = array; (void)copy;
        static_assert(std::is_copy_constructible<Array>::value, "");
        TEST_NOT_COPY_ASSIGNABLE(Array);
    }

    // Make sure we can implicitly copy a std::array of a non-trivially copyable type
    {
        typedef std::array<NonTrivialCopy, 0> Array;
        Array array = {};
        Array copy = array;
        copy = array;
        static_assert(std::is_copy_constructible<Array>::value, "");
    }
    {
        typedef std::array<NonTrivialCopy, 1> Array;
        Array array = {};
        Array copy = array;
        copy = array;
        static_assert(std::is_copy_constructible<Array>::value, "");
    }
    {
        typedef std::array<NonTrivialCopy, 2> Array;
        Array array = {};
        Array copy = array;
        copy = array;
        static_assert(std::is_copy_constructible<Array>::value, "");
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
