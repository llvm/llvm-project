//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivial

#include <type_traits>
#include "test_macros.h"

template <class T>
void test_is_trivial()
{
    static_assert( std::is_trivial<T>::value, "");
    static_assert( std::is_trivial<const T>::value, "");
    static_assert( std::is_trivial<volatile T>::value, "");
    static_assert( std::is_trivial<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_trivial_v<T>, "");
    static_assert( std::is_trivial_v<const T>, "");
    static_assert( std::is_trivial_v<volatile T>, "");
    static_assert( std::is_trivial_v<const volatile T>, "");
#endif
}

template <class T>
void test_is_not_trivial()
{
    static_assert(!std::is_trivial<T>::value, "");
    static_assert(!std::is_trivial<const T>::value, "");
    static_assert(!std::is_trivial<volatile T>::value, "");
    static_assert(!std::is_trivial<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!std::is_trivial_v<T>, "");
    static_assert(!std::is_trivial_v<const T>, "");
    static_assert(!std::is_trivial_v<volatile T>, "");
    static_assert(!std::is_trivial_v<const volatile T>, "");
#endif
}

struct A {};

class B
{
public:
    B();
};

int main(int, char**)
{
    test_is_trivial<int> ();
    test_is_trivial<A> ();

    test_is_not_trivial<int&> ();
    test_is_not_trivial<volatile int&> ();
    test_is_not_trivial<B> ();

  return 0;
}
