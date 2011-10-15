//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_move_constructible

#include <type_traits>

template <class T, bool Result>
void test_is_move_constructible()
{
    static_assert(std::is_move_constructible<T>::value == Result, "");
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

int main()
{
    test_is_move_constructible<char[3], false>();
    test_is_move_constructible<char[], false>();
    test_is_move_constructible<void, false>();
    test_is_move_constructible<Abstract, false>();

    test_is_move_constructible<A, true>();
    test_is_move_constructible<int&, true>();
    test_is_move_constructible<Union, true>();
    test_is_move_constructible<Empty, true>();
    test_is_move_constructible<int, true>();
    test_is_move_constructible<double, true>();
    test_is_move_constructible<int*, true>();
    test_is_move_constructible<const int*, true>();
    test_is_move_constructible<NotEmpty, true>();
    test_is_move_constructible<bit_zero, true>();
    test_is_move_constructible<B, true>();
}
