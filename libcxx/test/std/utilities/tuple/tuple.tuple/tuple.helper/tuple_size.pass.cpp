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

// UNSUPPORTED: c++98, c++03

#include <tuple>
#include <type_traits>

template <class T, std::size_t N>
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
}

int main(int, char**)
{
    test<std::tuple<>, 0>();
    test<std::tuple<int>, 1>();
    test<std::tuple<char, int>, 2>();
    test<std::tuple<char, char*, int>, 3>();

  return 0;
}
