//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// tuple_element<I, pair<T1, T2> >::type

#include <type_traits>
#include <utility>

template <class T1, class T2>
void test()
{
    {
    typedef T1 Exp1;
    typedef T2 Exp2;
    typedef std::pair<T1, T2> P;
    static_assert((std::is_same<typename std::tuple_element<0, P>::type, Exp1>::value), "");
    static_assert((std::is_same<typename std::tuple_element<1, P>::type, Exp2>::value), "");
    }
    {
    typedef T1 const Exp1;
    typedef T2 const Exp2;
    typedef std::pair<T1, T2> const P;
    static_assert((std::is_same<typename std::tuple_element<0, P>::type, Exp1>::value), "");
    static_assert((std::is_same<typename std::tuple_element<1, P>::type, Exp2>::value), "");
    }
    {
    typedef T1 volatile Exp1;
    typedef T2 volatile Exp2;
    typedef std::pair<T1, T2> volatile P;
    static_assert((std::is_same<typename std::tuple_element<0, P>::type, Exp1>::value), "");
    static_assert((std::is_same<typename std::tuple_element<1, P>::type, Exp2>::value), "");
    }
    {
    typedef T1 const volatile Exp1;
    typedef T2 const volatile Exp2;
    typedef std::pair<T1, T2> const volatile P;
    static_assert((std::is_same<typename std::tuple_element<0, P>::type, Exp1>::value), "");
    static_assert((std::is_same<typename std::tuple_element<1, P>::type, Exp2>::value), "");
    }
}

int main(int, char**)
{
    test<int, short>();
    test<int*, char>();

  return 0;
}
