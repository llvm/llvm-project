//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// tuple_size<pair<T1, T2> >::value

#include <utility>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::pair<int, short> P1;
        static_assert((std::tuple_size<P1>::value == 2), "");
    }
    {
        typedef std::pair<int, short> const P1;
        static_assert((std::tuple_size<P1>::value == 2), "");
    }
    {
        typedef std::pair<int, short> volatile P1;
        static_assert((std::tuple_size<P1>::value == 2), "");
    }
    {
        typedef std::pair<int, short> const volatile P1;
        static_assert((std::tuple_size<P1>::value == 2), "");
    }

  return 0;
}
