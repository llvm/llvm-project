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
//   void
//   sort(Iter first, Iter last, Compare comp);

#include <algorithm>
#include <cassert>
#include <functional>

#include "test_macros.h"
#include "test_iterators.h"
#include "MoveOnly.h"

const int LargeN = 128;

template<int N, class T, class Iter>
TEST_CONSTEXPR_CXX20 bool test()
{
    int orig[N] = {};
    unsigned x = 1;
    for (int i=0; i < N; ++i) {
        x = (x * 1664525) + 1013904223;
        orig[i] = x % 1000;
    }
    T work[N] = {};
    std::copy(orig, orig+N, work);
    std::sort(Iter(work), Iter(work+N), std::greater<T>());
    assert(std::is_sorted(work, work+N, std::greater<T>()));
    assert(std::is_permutation(work, work+N, orig));

    return true;
}

template<int N, class T, class Iter>
TEST_CONSTEXPR_CXX20 bool test_pointers()
{
    T data[N] = {};
    T *orig[N] = {};
    unsigned x = 1;
    for (int i=0; i < N; ++i) {
        orig[i] = &data[x % 258];
    }
    T *work[N] = {};
    std::copy(orig, orig+N, work);
    std::sort(Iter(work), Iter(work+N), std::greater<T*>());
    assert(std::is_sorted(work, work+N, std::greater<T*>()));
    assert(std::is_permutation(work, work+N, orig));

    return true;
}

int main(int, char**)
{
    test<7, int, int*>();
    test<7, int, random_access_iterator<int*> >();
    test<LargeN, int, int*>();
    test<LargeN, int, random_access_iterator<int*> >();

#if TEST_STD_VER >= 11
    test<7, MoveOnly, MoveOnly*>();
    test<7, MoveOnly, random_access_iterator<MoveOnly*> >();
    test<LargeN, MoveOnly, MoveOnly*>();
    test<LargeN, MoveOnly, random_access_iterator<MoveOnly*> >();
#endif

    test_pointers<17, char, char**>();
    test_pointers<17, char, random_access_iterator<char**> >();
    test_pointers<17, const char, const char**>();
    test_pointers<17, const char, random_access_iterator<const char**> >();
    test_pointers<17, int, int**>();
    test_pointers<17, int, random_access_iterator<int**> >();

#if TEST_STD_VER >= 20
    test<7, int, contiguous_iterator<int*>>();
    test<LargeN, int, contiguous_iterator<int*>>();
    test<7, MoveOnly, contiguous_iterator<MoveOnly*>>();
    test<LargeN, MoveOnly, contiguous_iterator<MoveOnly*>>();
    test_pointers<17, char, contiguous_iterator<char**>>();
    test_pointers<17, const char, contiguous_iterator<const char**>>();
    test_pointers<17, int, contiguous_iterator<int**>>();

    static_assert(test<7, int, int*>());
    static_assert(test<7, int, random_access_iterator<int*>>());
    static_assert(test<7, int, contiguous_iterator<int*>>());
    static_assert(test<LargeN, int, int*>());
    static_assert(test<LargeN, int, random_access_iterator<int*>>());
    static_assert(test<LargeN, int, contiguous_iterator<int*>>());

    static_assert(test<7, MoveOnly, MoveOnly*>());
    static_assert(test<7, MoveOnly, random_access_iterator<MoveOnly*>>());
    static_assert(test<7, MoveOnly, contiguous_iterator<MoveOnly*>>());
    static_assert(test<LargeN, MoveOnly, MoveOnly*>());
    static_assert(test<LargeN, MoveOnly, random_access_iterator<MoveOnly*>>());
    static_assert(test<LargeN, MoveOnly, contiguous_iterator<MoveOnly*>>());

    static_assert(test_pointers<17, char, char**>());
    static_assert(test_pointers<17, char, random_access_iterator<char**>>());
    static_assert(test_pointers<17, char, contiguous_iterator<char**>>());
    static_assert(test_pointers<17, const char, const char**>());
    static_assert(test_pointers<17, const char, random_access_iterator<const char**>>());
    static_assert(test_pointers<17, const char, contiguous_iterator<const char**>>());
    static_assert(test_pointers<17, int, int**>());
    static_assert(test_pointers<17, int, random_access_iterator<int**>>());
    static_assert(test_pointers<17, int, contiguous_iterator<int**>>());
#endif

    return 0;
}
