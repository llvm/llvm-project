//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_standard_layout

#include <type_traits>

template <class T1, class T2>
struct pair
{
    T1 first;
    T2 second;
};

int main()
{
    static_assert( std::is_standard_layout<int>::value, "");
    static_assert( std::is_standard_layout<int[3]>::value, "");
    static_assert(!std::is_standard_layout<int&>::value, "");
    static_assert(!std::is_standard_layout<volatile int&>::value, "");
    static_assert(( std::is_standard_layout<pair<int, double> >::value), "");
}
