//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template<class U, class V> pair& operator=(const pair<U, V>& p);

#include <utility>
#include <cassert>

#include "test_macros.h"
#if TEST_STD_VER >= 11
#include "archetypes.hpp"
#endif

int main(int, char**)
{
    {
        typedef std::pair<int, short> P1;
        typedef std::pair<double, long> P2;
        P1 p1(3, static_cast<short>(4));
        P2 p2;
        p2 = p1;
        assert(p2.first == 3);
        assert(p2.second == 4);
    }
#if TEST_STD_VER >= 11
    {
       using C = TestTypes::TestType;
       using P = std::pair<int, C>;
       using T = std::pair<long, C>;
       const T t(42, -42);
       P p(101, 101);
       C::reset_constructors();
       p = t;
       assert(C::constructed == 0);
       assert(C::assigned == 1);
       assert(C::copy_assigned == 1);
       assert(C::move_assigned == 0);
       assert(p.first == 42);
       assert(p.second.value == -42);
    }
#endif

  return 0;
}
