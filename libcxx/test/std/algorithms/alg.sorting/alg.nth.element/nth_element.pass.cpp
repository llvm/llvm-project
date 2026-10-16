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
//   nth_element(Iter first, Iter nth, Iter last);

#include <algorithm>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"
#include "MoveOnly.h"

template<class T, class Iter>
TEST_CONSTEXPR_CXX20 bool test()
{
    int orig[15] = {3,1,4,1,5, 9,2,6,5,3, 5,8,9,7,9};
    T work[15] = {3,1,4,1,5, 9,2,6,5,3, 5,8,9,7,9};
    for (int n = 0; n < 15; ++n) {
        for (int m = 0; m < n; ++m) {
            std::nth_element(Iter(work), Iter(work+m), Iter(work+n));
            assert(std::is_permutation(work, work+n, orig));
            // No element to m's left is greater than m.
            for (int i = 0; i < m; ++i) {
                assert(!(work[i] > work[m]));
            }
            // No element to m's right is less than m.
            for (int i = m; i < n; ++i) {
                assert(!(work[i] < work[m]));
            }
            std::copy(orig, orig+15, work);
        }
    }

    {
        T input[] = {3,1,4,1,5,9,2};
        std::nth_element(Iter(input), Iter(input+4), Iter(input+7));
        assert(input[4] == 4);
        assert(input[5] + input[6] == 5 + 9);
    }

    {
        T input[] = {0, 1, 2, 3, 4, 5, 7, 6};
        std::nth_element(Iter(input), Iter(input + 6), Iter(input + 8));
        assert(input[6] == 6);
        assert(input[7] == 7);
    }

    {
        T input[] = {1, 0, 2, 3, 4, 5, 6, 7};
        std::nth_element(Iter(input), Iter(input + 1), Iter(input + 8));
        assert(input[0] == 0);
        assert(input[1] == 1);
    }

    return true;
}

#if TEST_STD_VER >= 11
TEST_CONSTEXPR_CXX20 void test_pattern(const std::vector<int>& input) {
  std::vector<int> sorted = input;
  std::sort(sorted.begin(), sorted.end());
  int n           = static_cast<int>(input.size());
  int positions[] = {0, n / 4, n / 2, n - 2, n - 1};
  for (int nth : positions) {
    std::vector<int> v = input;
    std::nth_element(v.begin(), v.begin() + nth, v.end());
    assert(v[nth] == sorted[nth]);
    for (int i = 0; i != nth; ++i)
      assert(v[i] <= v[nth]);
    for (int i = nth + 1; i != n; ++i)
      assert(v[i] >= v[nth]);
  }
}

// Inputs that make a median-of-3 pivot do badly.
TEST_CONSTEXPR_CXX20 bool test_patterns(int n) {
  std::vector<int> v(n);
  for (int i = 0; i != n; ++i)
    v[i] = i < n / 2 ? i : n - i; // pipe organ
  test_pattern(v);
  for (int i = 0; i != n; ++i)
    v[i] = i % 64; // sawtooth
  test_pattern(v);
  for (int i = 0; i != n; ++i)
    v[i] = (i * 7919) % 16; // few distinct values
  test_pattern(v);
  return true;
}
#endif

int main(int, char**)
{
    test<int, random_access_iterator<int*> >();
    test<int, int*>();

#if TEST_STD_VER >= 11
    test<MoveOnly, random_access_iterator<MoveOnly*>>();
    test<MoveOnly, MoveOnly*>();
    test_patterns(4096);
#endif

#if TEST_STD_VER >= 20
    static_assert(test<int, random_access_iterator<int*>>());
    static_assert(test<int, int*>());
    static_assert(test<MoveOnly, random_access_iterator<MoveOnly*>>());
    static_assert(test<MoveOnly, MoveOnly*>());
    static_assert(test_patterns(256));
#endif

    return 0;
}
