//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivially_copyable

#include <type_traits>
#include <cassert>

struct A
{
    int i_;
};

struct B
{
    int i_;
    ~B() {assert(i_ == 0);}
};

class C
{
public:
    C();
};

int main()
{
    static_assert( std::is_trivially_copyable<int>::value, "");
    static_assert( std::is_trivially_copyable<const int>::value, "");
    static_assert(!std::is_trivially_copyable<int&>::value, "");
    static_assert( std::is_trivially_copyable<A>::value, "");
    static_assert( std::is_trivially_copyable<const A>::value, "");
    static_assert(!std::is_trivially_copyable<const A&>::value, "");
    static_assert(!std::is_trivially_copyable<B>::value, "");
    static_assert( std::is_trivially_copyable<C>::value, "");
}
