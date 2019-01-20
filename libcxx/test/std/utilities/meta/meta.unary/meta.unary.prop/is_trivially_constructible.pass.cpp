//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// template <class T, class... Args>
//   struct is_trivially_constructible;

#include <type_traits>
#include "test_macros.h"

template <class T>
void test_is_trivially_constructible()
{
    static_assert(( std::is_trivially_constructible<T>::value), "");
#if TEST_STD_VER > 14
    static_assert(( std::is_trivially_constructible_v<T>), "");
#endif
}

template <class T, class A0>
void test_is_trivially_constructible()
{
    static_assert(( std::is_trivially_constructible<T, A0>::value), "");
#if TEST_STD_VER > 14
    static_assert(( std::is_trivially_constructible_v<T, A0>), "");
#endif
}

template <class T>
void test_is_not_trivially_constructible()
{
    static_assert((!std::is_trivially_constructible<T>::value), "");
#if TEST_STD_VER > 14
    static_assert((!std::is_trivially_constructible_v<T>), "");
#endif
}

template <class T, class A0>
void test_is_not_trivially_constructible()
{
    static_assert((!std::is_trivially_constructible<T, A0>::value), "");
#if TEST_STD_VER > 14
    static_assert((!std::is_trivially_constructible_v<T, A0>), "");
#endif
}

template <class T, class A0, class A1>
void test_is_not_trivially_constructible()
{
    static_assert((!std::is_trivially_constructible<T, A0, A1>::value), "");
#if TEST_STD_VER > 14
    static_assert((!std::is_trivially_constructible_v<T, A0, A1>), "");
#endif
}

struct A
{
    explicit A(int);
    A(int, double);
};

int main()
{
    test_is_trivially_constructible<int> ();
    test_is_trivially_constructible<int, const int&> ();

    test_is_not_trivially_constructible<A, int> ();
    test_is_not_trivially_constructible<A, int, double> ();
    test_is_not_trivially_constructible<A> ();
}
