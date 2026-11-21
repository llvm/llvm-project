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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
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

    return true;
}

#if TEST_STD_VER >= 11
// McIlroy's "A Killer Adversary for Quicksort", also used by sort.pass.cpp. The value of an element is only decided
// when it gets compared, in a way that makes every pivot a bad one.
struct AdversaryComparator {
  TEST_CONSTEXPR_CXX20 AdversaryComparator(int n, std::vector<int>& values, std::size_t& comparisons)
      : gas_(n - 1), values_(values), comparisons_(comparisons) {
    values_.assign(n, gas_);
  }

  TEST_CONSTEXPR_CXX20 bool operator()(int x, int y) {
    ++comparisons_;
    if (values_[x] == gas_ && values_[y] == gas_) {
      if (x == candidate_)
        values_[x] = solid_++;
      else
        values_[y] = solid_++;
    }
    if (values_[x] == gas_)
      candidate_ = x;
    else if (values_[y] == gas_)
      candidate_ = y;
    return values_[x] < values_[y];
  }

private:
  int gas_;
  std::vector<int>& values_;
  std::size_t& comparisons_;
  int candidate_ = 0;
  int solid_     = 0;
};

TEST_CONSTEXPR_CXX20 void test_adversary(int n, int nth) {
  std::vector<int> indices(n);
  for (int i = 0; i != n; ++i)
    indices[i] = i;
  std::vector<int> values;
  std::size_t comparisons = 0;
  std::nth_element(indices.begin(), indices.begin() + nth, indices.end(), AdversaryComparator(n, values, comparisons));

  for (int i = 0; i != nth; ++i)
    assert(values[indices[i]] <= values[indices[nth]]);
  for (int i = nth + 1; i != n; ++i)
    assert(values[indices[i]] >= values[indices[nth]]);

  // Plain quickselect needs about n * n / 4 comparisons on this input.
  std::size_t log2_n = 0;
  for (int m = n; m > 1; m /= 2)
    ++log2_n;
  LIBCPP_ASSERT(comparisons <= 8 * static_cast<std::size_t>(n) * log2_n);
}

TEST_CONSTEXPR_CXX20 bool test_adversaries(int n) {
  test_adversary(n, 0);
  test_adversary(n, n / 2);
  test_adversary(n, n - 2);
  test_adversary(n, n - 1);
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
    test_adversaries(10000);
#endif

#if TEST_STD_VER >= 20
    static_assert(test<int, random_access_iterator<int*>>());
    static_assert(test<int, int*>());
    static_assert(test<MoveOnly, random_access_iterator<MoveOnly*>>());
    static_assert(test<MoveOnly, MoveOnly*>());
    static_assert(test_adversaries(100));
#endif

    return 0;
}
