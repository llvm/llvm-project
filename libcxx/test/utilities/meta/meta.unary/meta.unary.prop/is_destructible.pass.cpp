//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_destructible

#include <type_traits>

template <class T, bool Result>
void test_is_destructible()
{
    static_assert( std::is_destructible<T>::value == Result, "");
    static_assert( std::is_destructible<const T>::value == Result, "");
    static_assert( std::is_destructible<volatile T>::value == Result, "");
    static_assert( std::is_destructible<const volatile T>::value == Result, "");
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
    test_is_destructible<void, false>();
    test_is_destructible<A, true>();
    test_is_destructible<Abstract, false>();
    test_is_destructible<NotEmpty, false>();
    test_is_destructible<int&, true>();
    test_is_destructible<Union, true>();
    test_is_destructible<Empty, true>();
    test_is_destructible<int, true>();
    test_is_destructible<double, true>();
    test_is_destructible<int*, true>();
    test_is_destructible<const int*, true>();
    test_is_destructible<char[3], true>();
    test_is_destructible<bit_zero, true>();
}
