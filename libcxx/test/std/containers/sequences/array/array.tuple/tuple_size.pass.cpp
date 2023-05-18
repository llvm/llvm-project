//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// tuple_size<array<T, N> >::value

#include <array>

#include "test_macros.h"

template <class T, std::size_t N>
void test()
{
    {
    typedef std::array<T, N> C;
    static_assert((std::tuple_size<C>::value == N), "");
    }
    {
    typedef std::array<T const, N> C;
    static_assert((std::tuple_size<C>::value == N), "");
    }
    {
    typedef std::array<T volatile, N> C;
    static_assert((std::tuple_size<C>::value == N), "");
    }
    {
    typedef std::array<T const volatile, N> C;
    static_assert((std::tuple_size<C>::value == N), "");
    }
}

int main(int, char**)
{
    test<double, 0>();
    test<double, 3>();
    test<double, 5>();

  return 0;
}
