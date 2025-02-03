//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class RandomAccessIterator>
//     constexpr void               // constexpr since C++26
//     stable_sort(RandomAccessIterator first, RandomAccessIterator last);

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=200000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=200000000

#include <algorithm>
#include <cassert>
#include <iterator>
#include <random>
#include <vector>

#include "count_new.h"
#include "test_macros.h"

template <class Iterator>
TEST_CONSTEXPR_CXX26 void test_all_permutations(Iterator first, Iterator last) {
  using T = typename std::iterator_traits<Iterator>::value_type;

  do {
    std::vector<T> save(first, last);
    std::stable_sort(save.begin(), save.end());
    assert(std::is_sorted(save.begin(), save.end()));
  } while (std::next_permutation(first, last));
}

template <class Iterator>
TEST_CONSTEXPR_CXX26 void test_sort_exhaustive_impl(Iterator first, Iterator last, int start, Iterator real_last) {
  using T = typename std::iterator_traits<Iterator>::value_type;

  for (Iterator i = last; i > first + start;) {
    *--i = static_cast<T>(start);
    if (first == i) {
      test_all_permutations(first, real_last);
    }
    if (start > 0)
      test_sort_exhaustive_impl(first, i, start - 1, real_last);
  }
}

template <class T>
TEST_CONSTEXPR_CXX26 void test_sort_exhaustive(int N) {
  std::vector<T> vec;
  vec.resize(N);
  for (int i = 0; i < N; ++i) {
    test_sort_exhaustive_impl(vec.begin(), vec.end(), i, vec.end());
  }
}

template <class T>
TEST_CONSTEXPR_CXX26 std::vector<T> generate_sawtooth(int N, int M) {
  // Populate a sequence of length N with M different numbers
  std::vector<T> v;
  T x = 0;
  for (int i = 0; i < N; ++i) {
    v.push_back(x);
    if (++x == M)
      x = 0;
  }
  return v;
}

template <class T>
TEST_CONSTEXPR_CXX26 void test_larger_sorts(int N, int M) {
  assert(N != 0);
  assert(M != 0);

  // test saw tooth pattern
  {
    auto v = generate_sawtooth<T>(N, M);
    std::stable_sort(v.begin(), v.end());
    assert(std::is_sorted(v.begin(), v.end()));
  }

  // test random pattern
  {
    if (!TEST_IS_CONSTANT_EVALUATED) {
      auto v = generate_sawtooth<T>(N, M);
      std::mt19937 randomness;
      std::shuffle(v.begin(), v.end(), randomness);
      std::stable_sort(v.begin(), v.end());
      assert(std::is_sorted(v.begin(), v.end()));
    }
  }

  // test sorted pattern
  {
    auto v = generate_sawtooth<T>(N, M);
    std::sort(v.begin(), v.end());

    std::stable_sort(v.begin(), v.end());
    assert(std::is_sorted(v.begin(), v.end()));
  }

  // test reverse sorted pattern
  {
    auto v = generate_sawtooth<T>(N, M);
    std::sort(v.begin(), v.end());
    std::reverse(v.begin(), v.end());

    std::stable_sort(v.begin(), v.end());
    assert(std::is_sorted(v.begin(), v.end()));
  }

  // test swap ranges 2 pattern
  {
    auto v = generate_sawtooth<T>(N, M);
    std::sort(v.begin(), v.end());
    std::swap_ranges(v.begin(), v.begin() + (N / 2), v.begin() + (N / 2));

    std::stable_sort(v.begin(), v.end());
    assert(std::is_sorted(v.begin(), v.end()));
  }

  // test reverse swap ranges 2 pattern
  {
    auto v = generate_sawtooth<T>(N, M);
    std::sort(v.begin(), v.end());
    std::reverse(v.begin(), v.end());
    std::swap_ranges(v.begin(), v.begin() + (N / 2), v.begin() + (N / 2));

    std::stable_sort(v.begin(), v.end());
    assert(std::is_sorted(v.begin(), v.end()));
  }
}

template <class T>
TEST_CONSTEXPR_CXX26 void test_larger_sorts(int N) {
  test_larger_sorts<T>(N, 1);
  test_larger_sorts<T>(N, 2);
  test_larger_sorts<T>(N, 3);
  test_larger_sorts<T>(N, N / 2 - 1);
  test_larger_sorts<T>(N, N / 2);
  test_larger_sorts<T>(N, N / 2 + 1);
  test_larger_sorts<T>(N, N - 2);
  test_larger_sorts<T>(N, N - 1);
  test_larger_sorts<T>(N, N);
}

template <class T>
TEST_CONSTEXPR_CXX26 bool test() {
  // test null range
  {
    T value = 0;
    std::stable_sort(&value, &value);
  }

  // exhaustively test all possibilities up to length 8
  if (!TEST_IS_CONSTANT_EVALUATED) {
    test_sort_exhaustive<T>(1);
    test_sort_exhaustive<T>(2);
    test_sort_exhaustive<T>(3);
    test_sort_exhaustive<T>(4);
    test_sort_exhaustive<T>(5);
    test_sort_exhaustive<T>(6);
    test_sort_exhaustive<T>(7);
    test_sort_exhaustive<T>(8);
  }

  test_larger_sorts<T>(256);
  test_larger_sorts<T>(257);
  if (!TEST_IS_CONSTANT_EVALUATED) { // avoid blowing past constexpr evaluation limit
    test_larger_sorts<T>(499);
    test_larger_sorts<T>(500);
    test_larger_sorts<T>(997);
    test_larger_sorts<T>(1000);
    test_larger_sorts<T>(1009);
    test_larger_sorts<T>(1024);
    test_larger_sorts<T>(1031);
    test_larger_sorts<T>(2053);
  }

  // check that the algorithm works without memory
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!TEST_IS_CONSTANT_EVALUATED) {
    std::vector<T> vec(150, T(3));
    getGlobalMemCounter()->throw_after = 0;
    std::stable_sort(vec.begin(), vec.end());
  }
#endif

  return true;
}

int main(int, char**) {
  test<int>();
  test<float>();
#if TEST_STD_VER >= 26
  static_assert(test<int>());
  static_assert(test<float>());
#endif
  return 0;
}
