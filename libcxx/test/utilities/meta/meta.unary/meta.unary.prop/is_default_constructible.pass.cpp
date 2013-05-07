//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_default_constructible

#include <type_traits>

template <class T, bool Result>
void test_is_default_constructible()
{
    static_assert(std::is_default_constructible<T>::value == Result, "");
    static_assert(std::is_default_constructible<const T>::value == Result, "");
    static_assert(std::is_default_constructible<volatile T>::value == Result, "");
    static_assert(std::is_default_constructible<const volatile T>::value == Result, "");
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
    A();
};

class B
{
    B();
};

int main()
{
    test_is_default_constructible<void, false>();
    test_is_default_constructible<int&, false>();
    test_is_default_constructible<char[], false>();
    test_is_default_constructible<Abstract, false>();

    test_is_default_constructible<A, true>();
    test_is_default_constructible<B, false>();
    test_is_default_constructible<Union, true>();
    test_is_default_constructible<Empty, true>();
    test_is_default_constructible<int, true>();
    test_is_default_constructible<double, true>();
    test_is_default_constructible<int*, true>();
    test_is_default_constructible<const int*, true>();
    test_is_default_constructible<char[3], true>();
    test_is_default_constructible<NotEmpty, true>();
    test_is_default_constructible<bit_zero, true>();
}
