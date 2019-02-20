//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <tuple>

// template <class T> constexpr size_t tuple_size_v = tuple_size<T>::value;

#include <tuple>
#include <utility>
#include <array>

template <class Tuple, int Expect>
void test()
{
    static_assert(std::tuple_size_v<Tuple> == Expect, "");
    static_assert(std::tuple_size_v<Tuple> == std::tuple_size<Tuple>::value, "");
    static_assert(std::tuple_size_v<Tuple const> == std::tuple_size<Tuple>::value, "");
    static_assert(std::tuple_size_v<Tuple volatile> == std::tuple_size<Tuple>::value, "");
    static_assert(std::tuple_size_v<Tuple const volatile> == std::tuple_size<Tuple>::value, "");
}

int main(int, char**)
{
    test<std::tuple<>, 0>();

    test<std::tuple<int>, 1>();
    test<std::array<int, 1>, 1>();

    test<std::tuple<int, int>, 2>();
    test<std::pair<int, int>, 2>();
    test<std::array<int, 2>, 2>();

    test<std::tuple<int, int, int>, 3>();
    test<std::array<int, 3>, 3>();

  return 0;
}
