//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_move_constructible

#include <type_traits>
#include "test_macros.h"

template <class T>
void test_is_move_constructible()
{
    static_assert( std::is_move_constructible<T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_move_constructible_v<T>, "");
#endif
}

template <class T>
void test_is_not_move_constructible()
{
    static_assert(!std::is_move_constructible<T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!std::is_move_constructible_v<T>, "");
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

struct B
{
    B(B&&);
};

int main(int, char**)
{
    test_is_not_move_constructible<char[3]>();
    test_is_not_move_constructible<char[]>();
    test_is_not_move_constructible<void>();
    test_is_not_move_constructible<Abstract>();

    test_is_move_constructible<A>();
    test_is_move_constructible<int&>();
    test_is_move_constructible<Union>();
    test_is_move_constructible<Empty>();
    test_is_move_constructible<int>();
    test_is_move_constructible<double>();
    test_is_move_constructible<int*>();
    test_is_move_constructible<const int*>();
    test_is_move_constructible<NotEmpty>();
    test_is_move_constructible<bit_zero>();
    test_is_move_constructible<B>();

  return 0;
}
