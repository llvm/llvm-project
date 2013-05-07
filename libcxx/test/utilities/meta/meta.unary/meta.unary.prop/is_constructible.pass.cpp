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
//   struct is_constructible;

#include <type_traits>

struct A
{
    explicit A(int);
    A(int, double);
private:
    A(char);
};

int main()
{
    static_assert((std::is_constructible<int>::value), "");
    static_assert((std::is_constructible<int, const int>::value), "");
    static_assert((std::is_constructible<A, int>::value), "");
    static_assert((std::is_constructible<A, int, double>::value), "");
    static_assert((!std::is_constructible<A>::value), "");
    static_assert((!std::is_constructible<A, char>::value), "");
    static_assert((!std::is_constructible<A, void>::value), "");
    static_assert((!std::is_constructible<void>::value), "");
    static_assert((!std::is_constructible<int&>::value), "");
    static_assert(( std::is_constructible<int&, int&>::value), "");
}
