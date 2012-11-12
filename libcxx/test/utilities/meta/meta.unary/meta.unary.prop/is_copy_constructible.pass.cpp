//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_copy_constructible

#include <type_traits>

template <class T, bool Result>
void test_is_copy_constructible()
{
    static_assert(std::is_copy_constructible<T>::value == Result, "");
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

class B
{
    B(const B&);
};

int main()
{
    test_is_copy_constructible<char[3], false>();
    test_is_copy_constructible<char[], false>();
    test_is_copy_constructible<void, false>();
    test_is_copy_constructible<Abstract, false>();

    test_is_copy_constructible<A, true>();
    test_is_copy_constructible<B, false>();
    test_is_copy_constructible<int&, true>();
    test_is_copy_constructible<Union, true>();
    test_is_copy_constructible<Empty, true>();
    test_is_copy_constructible<int, true>();
    test_is_copy_constructible<double, true>();
    test_is_copy_constructible<int*, true>();
    test_is_copy_constructible<const int*, true>();
    test_is_copy_constructible<NotEmpty, true>();
    test_is_copy_constructible<bit_zero, true>();
}
