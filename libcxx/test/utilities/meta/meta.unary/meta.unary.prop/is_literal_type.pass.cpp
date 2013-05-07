//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_literal_type

#include <type_traits>

struct A
{
};

struct B
{
    B();
};

int main()
{
    static_assert( std::is_literal_type<int>::value, "");
    static_assert( std::is_literal_type<const int>::value, "");
    static_assert( std::is_literal_type<int&>::value, "");
    static_assert( std::is_literal_type<volatile int&>::value, "");
    static_assert( std::is_literal_type<A>::value, "");
    static_assert(!std::is_literal_type<B>::value, "");
}
