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
//     typename tuple_element<I, std::pair<T1, T2> >::type&
//     get(pair<T1, T2>&);

#include <utility>
#include <cassert>

#include "test_macros.h"

#if TEST_STD_VER > 11
struct S {
   std::pair<int, int> a;
   int k;
   constexpr S() : a{1,2}, k(std::get<0>(a)) {}
   };

constexpr std::pair<int, int> getP () { return { 3, 4 }; }
#endif

int main(int, char**)
{
    {
        typedef std::pair<int, short> P;
        P p(3, static_cast<short>(4));
        assert(std::get<0>(p) == 3);
        assert(std::get<1>(p) == 4);
        std::get<0>(p) = 5;
        std::get<1>(p) = 6;
        assert(std::get<0>(p) == 5);
        assert(std::get<1>(p) == 6);
    }

#if TEST_STD_VER > 11
    {
        static_assert(S().k == 1, "");
        static_assert(std::get<1>(getP()) == 4, "");
    }
#endif


  return 0;
}
