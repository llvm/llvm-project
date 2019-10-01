//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <size_t I, class... Types>
//   typename tuple_element<I, tuple<Types...> >::type const&
//   get(const tuple<Types...>& t);

// UNSUPPORTED: c++98, c++03

#include <tuple>
#include <string>
#include <cassert>

#include "test_macros.h"

struct Empty {};

int main(int, char**)
{
    {
        typedef std::tuple<int> T;
        const T t(3);
        assert(std::get<0>(t) == 3);
    }
    {
        typedef std::tuple<std::string, int> T;
        const T t("high", 5);
        assert(std::get<0>(t) == "high");
        assert(std::get<1>(t) == 5);
    }
#if TEST_STD_VER > 11
    {
        typedef std::tuple<double, int> T;
        constexpr T t(2.718, 5);
        static_assert(std::get<0>(t) == 2.718, "");
        static_assert(std::get<1>(t) == 5, "");
    }
    {
        typedef std::tuple<Empty> T;
        constexpr T t{Empty()};
        constexpr Empty e = std::get<0>(t);
        ((void)e); // Prevent unused warning
    }
#endif
    {
        typedef std::tuple<double&, std::string, int> T;
        double d = 1.5;
        const T t(d, "high", 5);
        assert(std::get<0>(t) == 1.5);
        assert(std::get<1>(t) == "high");
        assert(std::get<2>(t) == 5);
        std::get<0>(t) = 2.5;
        assert(std::get<0>(t) == 2.5);
        assert(std::get<1>(t) == "high");
        assert(std::get<2>(t) == 5);
        assert(d == 2.5);
    }

  return 0;
}
