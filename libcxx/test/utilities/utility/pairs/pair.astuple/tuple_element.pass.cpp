//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// tuple_element<I, pair<T1, T2> >::type

#include <utility>

int main()
{
    {
        typedef std::pair<int, short> P1;
        static_assert((std::is_same<std::tuple_element<0, P1>::type, int>::value), "");
        static_assert((std::is_same<std::tuple_element<1, P1>::type, short>::value), "");
    }
    {
        typedef std::pair<int*, char> P1;
        static_assert((std::is_same<std::tuple_element<0, P1>::type, int*>::value), "");
        static_assert((std::is_same<std::tuple_element<1, P1>::type, char>::value), "");
    }
}
