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
//   push_heap(Iter first, Iter last);

#include <algorithm>
#include <cassert>
#include <functional>

#include "test_macros.h"
#include "test_iterators.h"
#include "MoveOnly.h"

template<class T, class Iter>
TEST_CONSTEXPR_CXX20 bool test()
{
    T orig[15] = {3,1,4,1,5, 9,2,6,5,3, 5,8,9,7,9};
    T work[15] = {3,1,4,1,5, 9,2,6,5,3, 5,8,9,7,9};
    for (int i = 1; i < 15; ++i) {
        std::push_heap(Iter(work), Iter(work+i), std::greater<T>());
        assert(std::is_permutation(work, work+i, orig));
        assert(std::is_heap(work, work+i, std::greater<T>()));
    }

    {
        T input[] = {5, 3, 4, 1, 2};
        std::push_heap(Iter(input), Iter(input + 1), std::greater<T>()); assert(input[0] == 5);
        std::push_heap(Iter(input), Iter(input + 2), std::greater<T>()); assert(input[0] == 3);
        std::push_heap(Iter(input), Iter(input + 3), std::greater<T>()); assert(input[0] == 3);
        std::push_heap(Iter(input), Iter(input + 4), std::greater<T>()); assert(input[0] == 1);
        std::push_heap(Iter(input), Iter(input + 5), std::greater<T>()); assert(input[0] == 1);
        assert(std::is_heap(input, input + 5, std::greater<T>()));
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
