//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivially_copy_constructible

// XFAIL: gcc-4.9

#include <type_traits>
#include "test_macros.h"

template <class T>
void test_is_trivially_copy_constructible()
{
    static_assert( std::is_trivially_copy_constructible<T>::value, "");
    static_assert( std::is_trivially_copy_constructible<const T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_trivially_copy_constructible_v<T>, "");
    static_assert( std::is_trivially_copy_constructible_v<const T>, "");
#endif
}

template <class T>
void test_has_not_trivial_copy_constructor()
{
    static_assert(!std::is_trivially_copy_constructible<T>::value, "");
    static_assert(!std::is_trivially_copy_constructible<const T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!std::is_trivially_copy_constructible_v<T>, "");
    static_assert(!std::is_trivially_copy_constructible_v<const T>, "");
#endif
}

class Empty
{
};

class NotEmpty
{
public:
    virtual ~NotEmpty();
};

union Union {};

struct bit_zero
{
    int :  0;
};

class Abstract
{
public:
    virtual ~Abstract() = 0;
};

struct A
{
    A(const A&);
};

int main(int, char**)
{
    test_has_not_trivial_copy_constructor<void>();
    test_has_not_trivial_copy_constructor<A>();
    test_has_not_trivial_copy_constructor<Abstract>();
    test_has_not_trivial_copy_constructor<NotEmpty>();

    test_is_trivially_copy_constructible<int&>();
    test_is_trivially_copy_constructible<Union>();
    test_is_trivially_copy_constructible<Empty>();
    test_is_trivially_copy_constructible<int>();
    test_is_trivially_copy_constructible<double>();
    test_is_trivially_copy_constructible<int*>();
    test_is_trivially_copy_constructible<const int*>();
    test_is_trivially_copy_constructible<bit_zero>();

  return 0;
}
