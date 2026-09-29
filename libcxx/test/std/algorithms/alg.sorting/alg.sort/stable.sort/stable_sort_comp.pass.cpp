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
//   constexpr void stable_sort(Iter first, Iter last, Compare comp); // constexpr since C++26
//
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=200000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=200000000

#include <algorithm>
#include <functional>
#include <vector>
#include <random>
#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>

#include "test_macros.h"

struct indirect_less {
  template <class P>
  TEST_CONSTEXPR_CXX26 bool operator()(const P& x, const P& y) const {
    return *x < *y;
  }
};

struct first_only {
  TEST_CONSTEXPR_CXX26 bool operator()(const std::pair<int, int>& x, const std::pair<int, int>& y) const {
    return x.first < y.first;
  }
};

using Pair = std::pair<int, int>;

TEST_CONSTEXPR_CXX26 std::vector<Pair> generate_sawtooth(int N, int M) {
  std::vector<Pair> v(N);
  int x   = 0;
  int ver = 0;
  for (int i = 0; i < N; ++i) {
    v[i] = Pair(x, ver);
    if (++x == M) {
      x = 0;
      ++ver;
    }
  }
  return v;
}

TEST_CONSTEXPR_CXX26 bool test() {
  int const N = 1000;
  int const M = 10;

  // test sawtooth pattern
  {
    auto v = generate_sawtooth(N, M);
    std::stable_sort(v.begin(), v.end(), first_only());
    assert(std::is_sorted(v.begin(), v.end()));
  }

  // Test sorting a sequence where subsequences of elements are not sorted with <,
  // but everything is already sorted with respect to the first element. This ensures
  // that we don't change the order of "equivalent" elements.
  {
    if (!TEST_IS_CONSTANT_EVALUATED) {
      auto v = generate_sawtooth(N, M);
      std::mt19937 randomness;
      for (int i = 0; i < N - M; i += M) {
        std::shuffle(v.begin() + i, v.begin() + i + M, randomness);
      }
      std::stable_sort(v.begin(), v.end(), first_only());
      assert(std::is_sorted(v.begin(), v.end()));
    }
  }

#if TEST_STD_VER >= 11
  {
    std::vector<std::unique_ptr<int> > v(1000);
    for (int i = 0; static_cast<std::size_t>(i) < v.size(); ++i)
      v[i].reset(new int(i));
    std::stable_sort(v.begin(), v.end(), indirect_less());
    assert(std::is_sorted(v.begin(), v.end(), indirect_less()));
    assert(*v[0] == 0);
    assert(*v[1] == 1);
    assert(*v[2] == 2);
  }
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
