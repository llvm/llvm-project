//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template<class... Types>
//   tuple<VTypes...> make_tuple(Types&&... t);

// UNSUPPORTED: c++03

#include <tuple>
#include <functional>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX20
bool test()
{
    int i = 0;
    float j = 0;
    std::tuple<int, int&, float&> t =
        std::make_tuple(1, std::ref(i), std::ref(j));
    assert(std::get<0>(t) == 1);
    assert(std::get<1>(t) == 0);
    assert(std::get<2>(t) == 0);
    i = 2;
    j = 3.5;
    assert(std::get<0>(t) == 1);
    assert(std::get<1>(t) == 2);
    assert(std::get<2>(t) == 3.5);
    std::get<1>(t) = 0;
    std::get<2>(t) = 0;
    assert(i == 0);
    assert(j == 0);

    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER >= 20
    static_assert(test());
#endif

#if TEST_STD_VER > 11
    {
        constexpr auto t1 = std::make_tuple(0, 1, 3.14);
        constexpr int i1 = std::get<1>(t1);
        constexpr double d1 = std::get<2>(t1);
        static_assert (i1 == 1, "" );
        static_assert (d1 == 3.14, "" );
    }
#endif

    return 0;
}
