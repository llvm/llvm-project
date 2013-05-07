//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_destructible

#include <type_traits>

template <class T>
void test_is_nothrow_destructible()
{
    static_assert( std::is_nothrow_destructible<T>::value, "");
    static_assert( std::is_nothrow_destructible<const T>::value, "");
    static_assert( std::is_nothrow_destructible<volatile T>::value, "");
    static_assert( std::is_nothrow_destructible<const volatile T>::value, "");
}

template <class T>
void test_has_not_nothrow_destructor()
{
    static_assert(!std::is_nothrow_destructible<T>::value, "");
    static_assert(!std::is_nothrow_destructible<const T>::value, "");
    static_assert(!std::is_nothrow_destructible<volatile T>::value, "");
    static_assert(!std::is_nothrow_destructible<const volatile T>::value, "");
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
    ~A();
};

int main()
{
    test_has_not_nothrow_destructor<void>();
    test_has_not_nothrow_destructor<Abstract>();
    test_has_not_nothrow_destructor<NotEmpty>();

    test_is_nothrow_destructible<A>();
    test_is_nothrow_destructible<int&>();
    test_is_nothrow_destructible<Union>();
    test_is_nothrow_destructible<Empty>();
    test_is_nothrow_destructible<int>();
    test_is_nothrow_destructible<double>();
    test_is_nothrow_destructible<int*>();
    test_is_nothrow_destructible<const int*>();
    test_is_nothrow_destructible<char[3]>();
    test_is_nothrow_destructible<char[3]>();
    test_is_nothrow_destructible<bit_zero>();
}
