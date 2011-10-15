//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <array>

// tuple_element<I, array<T, N> >::type

#include <array>
#include <type_traits>

int main()
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        static_assert((std::is_same<std::tuple_element<0, C>::type, T>::value), "");
        static_assert((std::is_same<std::tuple_element<1, C>::type, T>::value), "");
        static_assert((std::is_same<std::tuple_element<2, C>::type, T>::value), "");
    }
    {
        typedef int T;
        typedef std::array<T, 3> C;
        static_assert((std::is_same<std::tuple_element<0, C>::type, T>::value), "");
        static_assert((std::is_same<std::tuple_element<1, C>::type, T>::value), "");
        static_assert((std::is_same<std::tuple_element<2, C>::type, T>::value), "");
    }
}
