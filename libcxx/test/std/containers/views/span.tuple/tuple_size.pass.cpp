//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <span>

// tuple_size<span<T, N> >::value

#include <span>

template <class T, std::size_t N>
void test()
{
    {
    typedef std::span<T, N> C;
    static_assert((std::tuple_size<C>::value == N), "");
    }
    {
    typedef std::span<T const, N> C;
    static_assert((std::tuple_size<C>::value == N), "");
    }
    {
    typedef std::span<T volatile, N> C;
    static_assert((std::tuple_size<C>::value == N), "");
    }
    {
    typedef std::span<T const volatile, N> C;
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
