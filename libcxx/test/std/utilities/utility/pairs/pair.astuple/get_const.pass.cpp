//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template<size_t I, class T1, class T2>
//     const typename tuple_element<I, std::pair<T1, T2> >::type&
//     get(const pair<T1, T2>&);

#include <utility>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::pair<int, short> P;
        const P p(3, static_cast<short>(4));
        assert(std::get<0>(p) == 3);
        assert(std::get<1>(p) == 4);
    }

#if TEST_STD_VER > 11
    {
        typedef std::pair<int, short> P;
        constexpr P p1(3, static_cast<short>(4));
        static_assert(std::get<0>(p1) == 3, "");
        static_assert(std::get<1>(p1) == 4, "");
    }
#endif

  return 0;
}
