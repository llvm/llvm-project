//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// template <class T, class... Args>
//   struct is_trivially_constructible;

#include <type_traits>

struct A
{
    explicit A(int);
    A(int, double);
};

int main()
{
    static_assert(( std::is_trivially_constructible<int>::value), "");
    static_assert(( std::is_trivially_constructible<int, const int&>::value), "");
    static_assert((!std::is_trivially_constructible<A, int>::value), "");
    static_assert((!std::is_trivially_constructible<A, int, double>::value), "");
    static_assert((!std::is_trivially_constructible<A>::value), "");
}
