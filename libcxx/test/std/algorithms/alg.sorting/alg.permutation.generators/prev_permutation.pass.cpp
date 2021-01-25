//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<BidirectionalIterator Iter>
//   requires ShuffleIterator<Iter>
//         && LessThanComparable<Iter::value_type>
//   constexpr bool  // constexpr in C++20
//   prev_permutation(Iter first, Iter last);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

#include <cstdio>

TEST_CONSTEXPR_CXX14 int factorial(int x)
{
    int r = 1;
    for (; x; --x)
        r *= x;
    return r;
}

template <class Iter>
TEST_CONSTEXPR_CXX20 bool
test()
{
    int ia[] = {6, 5, 4, 3, 2, 1};
    const int sa = sizeof(ia)/sizeof(ia[0]);
    int prev[sa];
    for (int e = 0; e <= sa; ++e)
    {
        int count = 0;
        bool x;
        do
        {
            std::copy(ia, ia+e, prev);
            x = std::prev_permutation(Iter(ia), Iter(ia+e));
            if (e > 1)
            {
                if (x)
                    assert(std::lexicographical_compare(ia, ia+e, prev, prev+e));
                else
                    assert(std::lexicographical_compare(prev, prev+e, ia, ia+e));
            }
            ++count;
        } while (x);
        assert(count == factorial(e));
    }
    return true;
}

int main(int, char**)
{
    test<bidirectional_iterator<int*> >();
    test<random_access_iterator<int*> >();
    test<int*>();

#if TEST_STD_VER >= 20
    static_assert(test<bidirectional_iterator<int*>>());
    static_assert(test<random_access_iterator<int*>>());
    static_assert(test<int*>());
#endif

    return 0;
}
