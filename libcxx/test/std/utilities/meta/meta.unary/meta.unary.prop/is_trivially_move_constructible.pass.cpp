//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivially_move_constructible

#include <type_traits>
#include "test_macros.h"

template <class T>
void test_is_trivially_move_constructible()
{
    static_assert( std::is_trivially_move_constructible<T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_trivially_move_constructible_v<T>, "");
#endif
}

template <class T>
void test_has_not_trivial_move_constructor()
{
    static_assert(!std::is_trivially_move_constructible<T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!std::is_trivially_move_constructible_v<T>, "");
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

#if TEST_STD_VER >= 11

struct MoveOnly1
{
    MoveOnly1(MoveOnly1&&);
};

struct MoveOnly2
{
    MoveOnly2(MoveOnly2&&) = default;
};

#endif

int main(int, char**)
{
    test_has_not_trivial_move_constructor<void>();
    test_has_not_trivial_move_constructor<A>();
    test_has_not_trivial_move_constructor<Abstract>();
    test_has_not_trivial_move_constructor<NotEmpty>();

    test_is_trivially_move_constructible<Union>();
    test_is_trivially_move_constructible<Empty>();
    test_is_trivially_move_constructible<int>();
    test_is_trivially_move_constructible<double>();
    test_is_trivially_move_constructible<int*>();
    test_is_trivially_move_constructible<const int*>();
    test_is_trivially_move_constructible<bit_zero>();

#if TEST_STD_VER >= 11
    static_assert(!std::is_trivially_move_constructible<MoveOnly1>::value, "");
    static_assert( std::is_trivially_move_constructible<MoveOnly2>::value, "");
#endif

  return 0;
}
