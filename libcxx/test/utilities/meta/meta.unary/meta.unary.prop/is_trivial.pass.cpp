//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivial

#include <type_traits>

struct A {};

class B
{
public:
    B();
};

int main()
{
    static_assert( std::is_trivial<int>::value, "");
    static_assert(!std::is_trivial<int&>::value, "");
    static_assert(!std::is_trivial<volatile int&>::value, "");
    static_assert( std::is_trivial<A>::value, "");
    static_assert(!std::is_trivial<B>::value, "");
}
