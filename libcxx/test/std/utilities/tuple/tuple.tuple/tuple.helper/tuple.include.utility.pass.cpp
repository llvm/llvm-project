//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... Types>
//   class tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };
//
//  LWG #2212 says that tuple_size and tuple_element must be
//     available after including <utility>

#include <cstddef>
#include <utility>
#include <type_traits>

#include "test_macros.h"

template <class T, std::size_t N, class U, size_t idx>
void test()
{
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   std::tuple_size<T> >::value), "");
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   std::tuple_size<const T> >::value), "");
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   std::tuple_size<volatile T> >::value), "");
    static_assert((std::is_base_of<std::integral_constant<std::size_t, N>,
                                   std::tuple_size<const volatile T> >::value), "");
    static_assert((std::is_same<typename std::tuple_element<idx, T>::type, U>::value), "");
    static_assert((std::is_same<typename std::tuple_element<idx, const T>::type, const U>::value), "");
    static_assert((std::is_same<typename std::tuple_element<idx, volatile T>::type, volatile U>::value), "");
    static_assert((std::is_same<typename std::tuple_element<idx, const volatile T>::type, const volatile U>::value), "");
}

int main(int, char**)
{
    test<std::pair<int, int>, 2, int, 0>();
    test<std::pair<int, int>, 2, int, 1>();
    test<std::pair<const int, int>, 2, int, 1>();
    test<std::pair<int, volatile int>, 2, volatile int, 1>();
    test<std::pair<char *, int>, 2, char *, 0>();
    test<std::pair<char *, int>, 2, int,    1>();

  return 0;
}
