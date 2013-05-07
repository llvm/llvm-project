//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_assignable

#include <type_traits>

struct A
{
};

struct B
{
    void operator=(A);
};

int main()
{
    static_assert(( std::is_assignable<int&, int&>::value), "");
    static_assert(( std::is_assignable<int&, int>::value), "");
    static_assert((!std::is_assignable<int, int&>::value), "");
    static_assert((!std::is_assignable<int, int>::value), "");
    static_assert(( std::is_assignable<int&, double>::value), "");
    static_assert(( std::is_assignable<B, A>::value), "");
    static_assert((!std::is_assignable<A, B>::value), "");
    static_assert((!std::is_assignable<void, const void>::value), "");
    static_assert((!std::is_assignable<const void, const void>::value), "");
    static_assert(( std::is_assignable<void*&, void*>::value), "");
    static_assert((!std::is_assignable<int(), int>::value), "");
}
