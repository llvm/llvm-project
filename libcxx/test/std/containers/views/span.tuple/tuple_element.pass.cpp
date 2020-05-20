//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <span>

// tuple_element<I, span<T, N> >::type

#include <span>

#include "test_macros.h"

template <class T, std::size_t N, std::size_t Idx>
void test()
{
    {
    typedef std::span<T, N> C;
    ASSERT_SAME_TYPE(typename std::tuple_element<Idx, C>::type, T);
    }
    {
    typedef std::span<T const, N> C;
    ASSERT_SAME_TYPE(typename std::tuple_element<Idx, C>::type, T const);
    }
    {
    typedef std::span<T volatile, N> C;
    ASSERT_SAME_TYPE(typename std::tuple_element<Idx, C>::type, T volatile);
    }
    {
    typedef std::span<T const volatile, N> C;
    ASSERT_SAME_TYPE(typename std::tuple_element<Idx, C>::type, T const volatile);
    }
}

int main(int, char**)
{
    test<double, 3, 0>();
    test<double, 3, 1>();
    test<double, 3, 2>();

    test<int, 5, 0>();
    test<int, 5, 1>();
    test<int, 5, 2>();
    test<int, 5, 3>();
    test<int, 5, 4>();

  return 0;
}
