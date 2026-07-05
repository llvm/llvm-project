//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter>
//   requires ShuffleIterator<Iter> && LessThanComparable<Iter::value_type>
//   constexpr void  // constexpr in C++20
//   pop_heap(Iter first, Iter last);

#include <algorithm>
#include <cassert>
#include <functional>

#include "test_macros.h"
#include "test_iterators.h"
#include "MoveOnly.h"

template<class T, class Iter>
TEST_CONSTEXPR_CXX20 bool test()
{
    T orig[15] = {9,6,9,5,5, 8,9,1,1,3, 5,3,4,7,2};
    T work[15] = {9,6,9,5,5, 8,9,1,1,3, 5,3,4,7,2};
    assert(std::is_heap(orig, orig+15));
    for (int i = 15; i >= 1; --i) {
        std::pop_heap(Iter(work), Iter(work+i));
        assert(std::is_heap(work, work+i-1));
        assert(std::max_element(work, work+i-1) == work);
        assert(std::is_permutation(work, work+15, orig));
    }
    assert(std::is_sorted(work, work+15));

    {
        T input[] = {5, 4, 1, 2, 3};
        assert(std::is_heap(input, input + 5));
        std::pop_heap(Iter(input), Iter(input + 5)); assert(input[4] == 5);
        std::pop_heap(Iter(input), Iter(input + 4)); assert(input[3] == 4);
        std::pop_heap(Iter(input), Iter(input + 3)); assert(input[2] == 3);
        std::pop_heap(Iter(input), Iter(input + 2)); assert(input[1] == 2);
        std::pop_heap(Iter(input), Iter(input + 1)); assert(input[0] == 1);
    }
    return true;
}

int main(int, char**)
{
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
