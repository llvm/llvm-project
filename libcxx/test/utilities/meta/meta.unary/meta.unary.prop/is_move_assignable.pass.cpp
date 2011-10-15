//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_move_assignable

#include <type_traits>

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

struct A
{
    A();
};

int main()
{
    static_assert(( std::is_move_assignable<int>::value), "");
    static_assert((!std::is_move_assignable<const int>::value), "");
    static_assert((!std::is_move_assignable<int[]>::value), "");
    static_assert((!std::is_move_assignable<int[3]>::value), "");
    static_assert((!std::is_move_assignable<int[3]>::value), "");
    static_assert((!std::is_move_assignable<void>::value), "");
    static_assert(( std::is_move_assignable<A>::value), "");
    static_assert(( std::is_move_assignable<bit_zero>::value), "");
    static_assert(( std::is_move_assignable<Union>::value), "");
    static_assert(( std::is_move_assignable<NotEmpty>::value), "");
    static_assert(( std::is_move_assignable<Empty>::value), "");
}
