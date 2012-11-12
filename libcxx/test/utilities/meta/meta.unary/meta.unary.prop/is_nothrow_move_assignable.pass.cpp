//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// has_nothrow_move_assign

#include <type_traits>

template <class T, bool Result>
void test_has_nothrow_assign()
{
    static_assert(std::is_nothrow_move_assignable<T>::value == Result, "");
}

class Empty
{
};

struct NotEmpty
{
    virtual ~NotEmpty();
};

union Union {};

struct bit_zero
{
    int :  0;
};

struct A
{
    A& operator=(const A&);
};

int main()
{
    test_has_nothrow_assign<void, false>();
    test_has_nothrow_assign<A, false>();
    test_has_nothrow_assign<int&, true>();

    test_has_nothrow_assign<Union, true>();
    test_has_nothrow_assign<Empty, true>();
    test_has_nothrow_assign<int, true>();
    test_has_nothrow_assign<double, true>();
    test_has_nothrow_assign<int*, true>();
    test_has_nothrow_assign<const int*, true>();
    test_has_nothrow_assign<NotEmpty, true>();
    test_has_nothrow_assign<bit_zero, true>();
}
