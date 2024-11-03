//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <algorithm>

// template<class Iter>
//   void sort_heap(Iter first, Iter last);

#include <algorithm>
#include <cassert>
#include <random>

#include "test_macros.h"

struct Stats {
  int compared = 0;
  int copied   = 0;
  int moved    = 0;
} stats;

struct MyInt {
  int value;
  explicit MyInt(int xval) : value(xval) {}
  MyInt(const MyInt& other) : value(other.value) { ++stats.copied; }
  MyInt(MyInt&& other) : value(other.value) { ++stats.moved; }
  MyInt& operator=(const MyInt& other) {
    value = other.value;
    ++stats.copied;
    return *this;
  }
  MyInt& operator=(MyInt&& other) {
    value = other.value;
    ++stats.moved;
    return *this;
  }
  friend bool operator<(const MyInt& a, const MyInt& b) {
    ++stats.compared;
    return a.value < b.value;
  }
};

int main(int, char**) {
  constexpr int N = (1 << 20);
  std::vector<MyInt> v;
  v.reserve(N);
  std::mt19937 g;
  for (int i = 0; i < N; ++i) {
    v.emplace_back(i);
  }
  for (int logn = 10; logn <= 20; ++logn) {
    const int n = (1 << logn);
    auto first  = v.begin();
    auto last   = v.begin() + n;
    const int debug_elements = std::min(100, n);
    // Multiplier 2 because of comp(a,b) comp(b, a) checks.
    const int debug_comparisons = 2 * (debug_elements + 1) * debug_elements;
    std::shuffle(first, last, g);
    std::make_heap(first, last);
    // The exact stats of our current implementation are recorded here.
    stats = {};
    std::sort_heap(first, last);
    LIBCPP_ASSERT(stats.copied == 0);
    LIBCPP_ASSERT(stats.moved <= 2 * n + n * logn);
#if !_LIBCPP_ENABLE_DEBUG_MODE
    LIBCPP_ASSERT(stats.compared <= n * logn);
    (void)debug_comparisons;
#else
    LIBCPP_ASSERT(stats.compared <= 2 * n * logn + debug_comparisons);
#endif
    LIBCPP_ASSERT(std::is_sorted(first, last));
  }
  return 0;
}
