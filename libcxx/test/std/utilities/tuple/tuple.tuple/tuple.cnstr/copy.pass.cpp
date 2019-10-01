//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// tuple(const tuple& u) = default;

// UNSUPPORTED: c++98, c++03

#include <tuple>
#include <string>
#include <cassert>

#include "test_macros.h"

struct Empty {};

int main(int, char**)
{
    {
        typedef std::tuple<> T;
        T t0;
        T t = t0;
        ((void)t); // Prevent unused warning
    }
    {
        typedef std::tuple<int> T;
        T t0(2);
        T t = t0;
        assert(std::get<0>(t) == 2);
    }
    {
        typedef std::tuple<int, char> T;
        T t0(2, 'a');
        T t = t0;
        assert(std::get<0>(t) == 2);
        assert(std::get<1>(t) == 'a');
    }
    {
        typedef std::tuple<int, char, std::string> T;
        const T t0(2, 'a', "some text");
        T t = t0;
        assert(std::get<0>(t) == 2);
        assert(std::get<1>(t) == 'a');
        assert(std::get<2>(t) == "some text");
    }
#if TEST_STD_VER > 11
    {
        typedef std::tuple<int> T;
        constexpr T t0(2);
        constexpr T t = t0;
        static_assert(std::get<0>(t) == 2, "");
    }
    {
        typedef std::tuple<Empty> T;
        constexpr T t0;
        constexpr T t = t0;
        constexpr Empty e = std::get<0>(t);
        ((void)e); // Prevent unused warning
    }
#endif

  return 0;
}
