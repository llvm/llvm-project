//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_default_constructible

#include <type_traits>

template <class T>
void test_is_nothrow_default_constructible()
{
    static_assert( std::is_nothrow_default_constructible<T>::value, "");
    static_assert( std::is_nothrow_default_constructible<const T>::value, "");
    static_assert( std::is_nothrow_default_constructible<volatile T>::value, "");
    static_assert( std::is_nothrow_default_constructible<const volatile T>::value, "");
}

template <class T>
void test_has_not_nothrow_default_constructor()
{
    static_assert(!std::is_nothrow_default_constructible<T>::value, "");
    static_assert(!std::is_nothrow_default_constructible<const T>::value, "");
    static_assert(!std::is_nothrow_default_constructible<volatile T>::value, "");
    static_assert(!std::is_nothrow_default_constructible<const volatile T>::value, "");
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
    A();
};

int main()
{
    test_has_not_nothrow_default_constructor<void>();
    test_has_not_nothrow_default_constructor<int&>();
    test_has_not_nothrow_default_constructor<A>();

    test_is_nothrow_default_constructible<Union>();
    test_is_nothrow_default_constructible<Empty>();
    test_is_nothrow_default_constructible<int>();
    test_is_nothrow_default_constructible<double>();
    test_is_nothrow_default_constructible<int*>();
    test_is_nothrow_default_constructible<const int*>();
    test_is_nothrow_default_constructible<char[3]>();
    test_is_nothrow_default_constructible<bit_zero>();
}
