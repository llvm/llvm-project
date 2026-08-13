//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter, StrictWeakOrder<auto, Iter::value_type> Compare>
//   requires ShuffleIterator<Iter> && CopyConstructible<Compare>
//   constexpr void  // constexpr in C++20
//   nth_element(Iter first, Iter nth, Iter last, Compare comp);

#include "test_macros.h"

#include <algorithm>
#include <cassert>
#include <functional>
#if TEST_STD_VER >= 20
#  include <type_traits>
#endif

#include "test_iterators.h"
#include "MoveOnly.h"

template <class T, class Iter>
TEST_CONSTEXPR_CXX20 void test_small_ranges(int max_n) {
  for (int n = 0; n <= max_n; ++n) {
    int perm[8] = {};
    for (int i = 0; i < n; ++i)
      perm[i] = i;
    do {
      for (int m = 0; m <= n; ++m) {
        T work[8];
        for (int i = 0; i < n; ++i)
          work[i] = T(perm[i]);
        std::nth_element(Iter(work), Iter(work + m), Iter(work + n), std::greater<T>());
        if (m == n)
          continue;
        assert(work[m] == T(n - 1 - m));
        for (int i = 0; i < m; ++i)
          assert(!(work[i] < work[m]));
        for (int i = m; i < n; ++i)
          assert(!(work[i] > work[m]));
      }
    } while (std::next_permutation(perm, perm + n));
  }

  for (int n = 1; n <= max_n; ++n) {
    for (int mask = 0; mask < (1 << n); ++mask) {
      int orig[8] = {};
      for (int i = 0; i < n; ++i)
        orig[i] = (mask >> i) & 1;
      int sorted[8] = {};
      std::copy(orig, orig + n, sorted);
      std::sort(sorted, sorted + n, std::greater<int>());
      for (int m = 0; m < n; ++m) {
        T work[8];
        for (int i = 0; i < n; ++i)
          work[i] = T(orig[i]);
        std::nth_element(Iter(work), Iter(work + m), Iter(work + n), std::greater<T>());
        assert(work[m] == T(sorted[m]));
        for (int i = 0; i < m; ++i)
          assert(!(work[i] < work[m]));
        for (int i = m; i < n; ++i)
          assert(!(work[i] > work[m]));
      }
    }
  }
}

template<class T, class Iter>
TEST_CONSTEXPR_CXX20 bool test()
{
    int orig[15] = {3,1,4,1,5, 9,2,6,5,3, 5,8,9,7,9};
    T work[15] = {3,1,4,1,5, 9,2,6,5,3, 5,8,9,7,9};
    for (int n = 0; n < 15; ++n) {
        for (int m = 0; m < n; ++m) {
            std::nth_element(Iter(work), Iter(work+m), Iter(work+n), std::greater<T>());
            assert(std::is_permutation(work, work+n, orig));
            // No element to m's left is less than m.
            for (int i = 0; i < m; ++i) {
                assert(!(work[i] < work[m]));
            }
            // No element to m's right is greater than m.
            for (int i = m; i < n; ++i) {
                assert(!(work[i] > work[m]));
            }
            std::copy(orig, orig+15, work);
        }
    }

    {
        T input[] = {3,1,4,1,5,9,2};
        std::nth_element(Iter(input), Iter(input+4), Iter(input+7), std::greater<T>());
        assert(input[4] == 2);
        assert(input[5] + input[6] == 1 + 1);
    }

    {
        T input[] = {0, 1, 2, 3, 4, 5, 7, 6};
        std::nth_element(Iter(input), Iter(input + 6), Iter(input + 8), std::greater<T>());
        assert(input[6] == 1);
        assert(input[7] == 0);
    }

    {
        T input[] = {1, 0, 2, 3, 4, 5, 6, 7};
        std::nth_element(Iter(input), Iter(input + 1), Iter(input + 8), std::greater<T>());
        assert(input[0] == 7);
        assert(input[1] == 6);
    }

    // Exhaustive check of all permutations of small ranges, and of 0/1 inputs with
    // duplicates. Lower the size during constant evaluation to stay within constexpr
    // step limits.
    int max_n = 8;
    if (TEST_IS_CONSTANT_EVALUATED)
      max_n = 5;
    test_small_ranges<T, Iter>(max_n);

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
