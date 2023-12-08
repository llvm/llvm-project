//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<Iterator Iter1, Iterator Iter2>
//   requires HasSwap<Iter1::reference, Iter2::reference>
//   void
//   iter_swap(Iter1 a, Iter2 b);

#include <algorithm>
#include <cassert>

#include "test_macros.h"

#if TEST_STD_VER > 17
constexpr bool test_swap_constexpr()
{
    int i = 1;
    int j = 2;
    std::iter_swap(&i, &j);
    return i == 2 && j == 1;
}
#endif // TEST_STD_VER > 17

int main(int, char**)
{
    int i = 1;
    int j = 2;
    std::iter_swap(&i, &j);
    assert(i == 2);
    assert(j == 1);

#if TEST_STD_VER > 17
    static_assert(test_swap_constexpr());
#endif // TEST_STD_VER > 17

  return 0;
}
