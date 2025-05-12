//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <algorithm>

// template<class Iter>
//   void make_heap(Iter first, Iter last);

// This test ensures that equivalent elements are not moved or copied when the heap is created.

#include <algorithm>
#include <cassert>
#include <vector>
#include <utility>

struct Stats {
  int compared = 0;
  int copied   = 0;
  int moved    = 0;
} stats;

struct MyPair {
  std::pair<int, int> p;
  MyPair(int a, int b) : p{a, b} {}
  MyPair(const MyPair& other) : p(other.p) { ++stats.copied; }
  MyPair(MyPair&& other) : p(other.p) { ++stats.moved; }
  MyPair& operator=(const MyPair& other) {
    p = other.p;
    ++stats.copied;
    return *this;
  }
  MyPair& operator=(MyPair&& other) {
    p = other.p;
    ++stats.moved;
    return *this;
  }
  friend bool operator<(const MyPair& lhs, const MyPair& rhs) {
    ++stats.compared;
    return lhs.p.first < rhs.p.first;
  }
  friend bool operator==(const MyPair& lhs, const MyPair& rhs) { return lhs.p == rhs.p; }
};

int main(int, char**) {
  std::vector<MyPair> hp{{42, 1}, {42, 2}, {42, 3}, {42, 4}, {42, 5}, {42, 6}};
  std::vector<MyPair> original_hp = hp;

  stats = {};
  std::make_heap(hp.begin(), hp.end());

  assert(stats.copied == 0);
  assert(stats.moved == 0);
  assert(stats.compared == static_cast<int>(hp.size()) - 1);

  assert(hp == original_hp);
  assert(std::is_heap(hp.begin(), hp.end()));

  return 0;
}
