//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter, RandomAccessIterator RAIter>
//   requires ShuffleIterator<RAIter>
//         && OutputIterator<RAIter, InIter::reference>
//         && HasLess<InIter::value_type, RAIter::value_type>
//         && LessThanComparable<RAIter::value_type>
//   constexpr RAIter  // constexpr in C++20
//   partial_sort_copy(InIter first, InIter last, RAIter result_first, RAIter result_last);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "MoveOnly.h"

template<class T, class Iter, class OutIter>
TEST_CONSTEXPR_CXX20 bool test()
{
    int orig[15] = {3,1,4,1,5, 9,2,6,5,3, 5,8,9,7,9};
    T work[15] = {};
    for (int n = 0; n < 15; ++n) {
        for (int m = 0; m < 15; ++m) {
            OutIter it = std::partial_sort_copy(Iter(orig), Iter(orig+n), OutIter(work), OutIter(work+m));
            if (n <= m) {
                assert(it == OutIter(work+n));
                assert(std::is_permutation(OutIter(work), it, orig));
            } else {
                assert(it == OutIter(work+m));
            }
            assert(std::is_sorted(OutIter(work), it));
            if (it != OutIter(work)) {
                // At most m-1 elements in the input are less than the biggest element in the result.
                int count = 0;
                for (int i = m; i < n; ++i) {
                    count += (T(orig[i]) < *(it - 1));
                }
                assert(count < m);
            }
        }
    }

    {
        int input[] = {3, 4, 2, 5, 1};
        T output[] = {0, 0, 0};
        std::partial_sort_copy(Iter(input), Iter(input + 5), OutIter(output), OutIter(output + 3));
        assert(output[0] == 1);
        assert(output[1] == 2);
        assert(output[2] == 3);
    }
    return true;
}

int main(int, char**)
{
    int i = 42;
    int j = 75;
    std::partial_sort_copy(&i, &i, &j, &j);  // no-op
    assert(i == 42);
    assert(j == 75);

    test<int, random_access_iterator<int*>, random_access_iterator<int*> >();
    test<int, random_access_iterator<int*>, int*>();
    test<int, int*, random_access_iterator<int*> >();
    test<int, int*, int*>();

#if TEST_STD_VER >= 11
    test<MoveOnly, random_access_iterator<int*>, random_access_iterator<MoveOnly*>>();
    test<MoveOnly, random_access_iterator<int*>, MoveOnly*>();
    test<MoveOnly, int*, random_access_iterator<MoveOnly*>>();
    test<MoveOnly, int*, MoveOnly*>();
#endif

#if TEST_STD_VER >= 20
    static_assert(test<int, random_access_iterator<int*>, random_access_iterator<int*>>());
    static_assert(test<int, int*, random_access_iterator<int*>>());
    static_assert(test<int, random_access_iterator<int*>, int*>());
    static_assert(test<int, int*, int*>());
    static_assert(test<MoveOnly, random_access_iterator<int*>, random_access_iterator<MoveOnly*>>());
    static_assert(test<MoveOnly, random_access_iterator<int*>, MoveOnly*>());
    static_assert(test<MoveOnly, int*, random_access_iterator<MoveOnly*>>());
    static_assert(test<MoveOnly, int*, MoveOnly*>());
#endif

    return 0;
}
