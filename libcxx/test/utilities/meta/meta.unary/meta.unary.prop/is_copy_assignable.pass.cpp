//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_copy_assignable

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

class B
{
    B& operator=(const B&);
};

int main()
{
    static_assert(( std::is_copy_assignable<int>::value), "");
    static_assert((!std::is_copy_assignable<const int>::value), "");
    static_assert((!std::is_copy_assignable<int[]>::value), "");
    static_assert((!std::is_copy_assignable<int[3]>::value), "");
    static_assert(( std::is_copy_assignable<int&>::value), "");
    static_assert(( std::is_copy_assignable<A>::value), "");
    static_assert(( std::is_copy_assignable<bit_zero>::value), "");
    static_assert(( std::is_copy_assignable<Union>::value), "");
    static_assert(( std::is_copy_assignable<NotEmpty>::value), "");
    static_assert(( std::is_copy_assignable<Empty>::value), "");
    static_assert((!std::is_copy_assignable<B>::value), "");
    static_assert((!std::is_copy_assignable<void>::value), "");
}
