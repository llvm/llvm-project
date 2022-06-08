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
  int copied = 0;
  int moved = 0;
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

int main(int, char**)
{
  const int N = 100'000;
  std::vector<MyInt> v;
  v.reserve(N);
  std::mt19937 g;
  for (int i = 0; i < N; ++i)
    v.emplace_back(g());
  std::make_heap(v.begin(), v.end());

  // The exact stats of our current implementation are recorded here.
  // If something changes to make them go a bit up or down, that's probably fine,
  // and we can just update this test.
  // But if they suddenly leap upward, that's a bad thing.

  stats = {};
  std::sort_heap(v.begin(), v.end());
  assert(stats.copied == 0);
  assert(stats.moved == 1'764'997);
#ifndef _LIBCPP_ENABLE_DEBUG_MODE
  assert(stats.compared == 1'534'701);
#endif

  assert(std::is_sorted(v.begin(), v.end()));

  return 0;
}
