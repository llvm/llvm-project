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
//   make_heap(Iter first, Iter last);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "MoveOnly.h"

template<class T, class Iter>
TEST_CONSTEXPR_CXX20 bool test()
{
    int orig[15] = {3,1,4,1,5, 9,2,6,5,3, 5,8,9,7,9};
    T work[15] = {3,1,4,1,5, 9,2,6,5,3, 5,8,9,7,9};
    for (int n = 0; n < 15; ++n) {
        std::make_heap(Iter(work), Iter(work+n));
        assert(std::is_heap(work, work+n));
        assert(std::is_permutation(work, work+n, orig));
        std::copy(orig, orig+n, work);
    }

    {
        T input[] = {3, 4, 1, 2, 5};
        std::make_heap(Iter(input), Iter(input + 5));
        assert(std::is_heap(input, input + 5));
        std::pop_heap(input, input + 5); assert(input[4] == 5);
        std::pop_heap(input, input + 4); assert(input[3] == 4);
        std::pop_heap(input, input + 3); assert(input[2] == 3);
        std::pop_heap(input, input + 2); assert(input[1] == 2);
        std::pop_heap(input, input + 1); assert(input[0] == 1);
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
