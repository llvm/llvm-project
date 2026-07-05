//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter, StrictWeakOrder<auto, Iter::value_type> Compare>
//   requires ShuffleIterator<Iter>
//         && CopyConstructible<Compare>
//   constexpr void  // constexpr in C++20
//   partial_sort(Iter first, Iter middle, Iter last, Compare comp);

#include <algorithm>
#include <cassert>
#include <functional>

#include "test_macros.h"
#include "test_iterators.h"
#include "MoveOnly.h"

template<class T, class Iter>
TEST_CONSTEXPR_CXX20 bool test()
{
    int orig[15] = {3,1,4,1,5, 9,2,6,5,3, 5,8,9,7,9};
    T work[15] = {3,1,4,1,5, 9,2,6,5,3, 5,8,9,7,9};
    for (int n = 0; n < 15; ++n) {
        for (int m = 0; m <= n; ++m) {
            std::partial_sort(Iter(work), Iter(work+m), Iter(work+n), std::greater<T>());
            assert(std::is_sorted(work, work+m, std::greater<T>()));
            assert(std::is_permutation(work, work+n, orig));
            // No element in the unsorted portion is greater than any element in the sorted portion.
            for (int i = m; i < n; ++i) {
                assert(m == 0 || !(work[i] > work[m-1]));
            }
            std::copy(orig, orig+15, work);
        }
    }

    {
        T input[] = {3, 4, 2, 5, 1};
        std::partial_sort(Iter(input), Iter(input + 3), Iter(input + 5), std::greater<T>());
        assert(input[0] == 5);
        assert(input[1] == 4);
        assert(input[2] == 3);
        assert(input[3] + input[4] == 1 + 2);
    }
    return true;
}

int main(int, char**)
{
    int i = 42;
    std::partial_sort(&i, &i, &i, std::greater<int>());  // no-op
    assert(i == 42);

    test<int, random_access_iterator<int*> >();
    test<int, int*>();

#if TEST_STD_VER >= 11
    test<MoveOnly, random_access_iterator<MoveOnly*>>();
    test<MoveOnly, MoveOnly*>();
#endif

#if TEST_STD_VER >= 20
    static_assert(test<int, random_access_iterator<int*>>());
    static_assert(test<int, int*>());
    static_assert(test<MoveOnly, random_access_iterator<MoveOnly*>>());
    static_assert(test<MoveOnly, MoveOnly*>());
#endif

    return 0;
}
