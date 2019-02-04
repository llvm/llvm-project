//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivially_copy_assignable

// XFAIL: gcc-4.9

#include <type_traits>
#include "test_macros.h"

template <class T>
void test_has_trivially_copy_assignable()
{
    static_assert( std::is_trivially_copy_assignable<T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_trivially_copy_assignable_v<T>, "");
#endif
}

template <class T>
void test_has_not_trivially_copy_assignable()
{
    static_assert(!std::is_trivially_copy_assignable<T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!std::is_trivially_copy_assignable_v<T>, "");
#endif
}

class Empty
{
};

class NotEmpty
{
    virtual ~NotEmpty();
};

union Union {};

struct bit_zero
{
    int :  0;
};

class Abstract
{
    virtual ~Abstract() = 0;
};

struct A
{
    A& operator=(const A&);
};

int main(int, char**)
{
    test_has_trivially_copy_assignable<int&>();
    test_has_trivially_copy_assignable<Union>();
    test_has_trivially_copy_assignable<Empty>();
    test_has_trivially_copy_assignable<int>();
    test_has_trivially_copy_assignable<double>();
    test_has_trivially_copy_assignable<int*>();
    test_has_trivially_copy_assignable<const int*>();
    test_has_trivially_copy_assignable<bit_zero>();

    test_has_not_trivially_copy_assignable<void>();
    test_has_not_trivially_copy_assignable<A>();
    test_has_not_trivially_copy_assignable<NotEmpty>();
    test_has_not_trivially_copy_assignable<Abstract>();
    test_has_not_trivially_copy_assignable<const Empty>();


  return 0;
}
