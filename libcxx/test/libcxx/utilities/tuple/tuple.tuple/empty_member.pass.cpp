//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// UNSUPPORTED: c++98, c++03

// This is not a portable test

#include <tuple>

struct A {};

struct B {};

int main()
{
    {
        typedef std::tuple<int, A> T;
        static_assert((sizeof(T) == sizeof(int)), "");
    }
    {
        typedef std::tuple<A, int> T;
        static_assert((sizeof(T) == sizeof(int)), "");
    }
    {
        typedef std::tuple<A, int, B> T;
        static_assert((sizeof(T) == sizeof(int)), "");
    }
    {
        typedef std::tuple<A, B, int> T;
        static_assert((sizeof(T) == sizeof(int)), "");
    }
    {
        typedef std::tuple<int, A, B> T;
        static_assert((sizeof(T) == sizeof(int)), "");
    }
}
