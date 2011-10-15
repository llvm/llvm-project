//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// has_nothrow_move_constructor

#include <type_traits>

template <class T>
void test_is_nothrow_move_constructible()
{
    static_assert( std::is_nothrow_move_constructible<T>::value, "");
    static_assert( std::is_nothrow_move_constructible<const T>::value, "");
}

template <class T>
void test_has_not_nothrow_move_constructor()
{
    static_assert(!std::is_nothrow_move_constructible<T>::value, "");
    static_assert(!std::is_nothrow_move_constructible<const T>::value, "");
    static_assert(!std::is_nothrow_move_constructible<volatile T>::value, "");
    static_assert(!std::is_nothrow_move_constructible<const volatile T>::value, "");
}

class Empty
{
};

union Union {};

struct bit_zero
{
    int :  0;
};

struct A
{
    A(const A&);
};

int main()
{
    test_has_not_nothrow_move_constructor<void>();
    test_has_not_nothrow_move_constructor<A>();

    test_is_nothrow_move_constructible<int&>();
    test_is_nothrow_move_constructible<Union>();
    test_is_nothrow_move_constructible<Empty>();
    test_is_nothrow_move_constructible<int>();
    test_is_nothrow_move_constructible<double>();
    test_is_nothrow_move_constructible<int*>();
    test_is_nothrow_move_constructible<const int*>();
    test_is_nothrow_move_constructible<bit_zero>();
}
