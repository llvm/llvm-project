//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive

// <hive>

// template<class... Args> iterator emplace(Args&&... args);

#include <algorithm>
#include <cassert>
#include <concepts>
#include <hive>
#include <iterator>
#include <memory>
#include <vector>

#include "../helpers.h"
#include "Counter.h"
#include "MoveOnly.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

struct Pair {
  int a;
  int b;
  Pair(int x, int y) : a(x), b(y) {}
};

// The arguments are forwarded to the constructor; the result points to the new element.
template <class A>
void test_returns_the_new_element() {
  using H = std::hive<Pair, A>;
  H h;
  std::same_as<typename H::iterator> decltype(auto) it = h.emplace(1, 2);
  assert(it->a == 1);
  assert(it->b == 2);
  assert(h.size() == 1);
  assert(it == h.begin());
  assert(std::next(it) == h.end());
  check_invariant(h);

  std::same_as<typename H::iterator> decltype(auto) it2 = h.emplace(3, 4);
  assert(it2->a == 3);
  assert(it2->b == 4);
  assert(it2 != it);
  assert(it->a == 1);
  assert(h.size() == 2);
  check_invariant(h);
}

// A move-only type is emplaced from an rvalue and from its constructor arguments.
void test_move_only() {
  std::hive<MoveOnly> h;
  auto it = h.emplace(MoveOnly(5));
  assert(it->get() == 5);
  auto it2 = h.emplace(6);
  assert(it2->get() == 6);
  assert(h.size() == 2);
  check_invariant(h);
}

// Exactly one object is constructed per call.
void test_constructs_exactly_one_object() {
  {
    std::hive<Counter<int>> h;
    assert(Counter_base::gConstructed == 0);
    h.emplace(1);
    assert(Counter_base::gConstructed == 1);
    h.emplace(2);
    assert(Counter_base::gConstructed == 2);
    h.emplace(Counter<int>(3));
    assert(Counter_base::gConstructed == 3);
    assert(h.size() == 3);
  }
  assert(Counter_base::gConstructed == 0);
}

// The arguments may refer to an element of the hive.
template <class A>
void test_argument_from_this() {
  std::hive<int, A> h;
  h.emplace(7);
  h.emplace(*h.begin());
  h.insert(*h.begin());
  assert(h.size() == 3);
  assert(std::count(h.begin(), h.end(), 7) == 3);
  check_invariant(h);
}

// Many elements, over several blocks: all reachable, the capacity grows with the size.
template <class A>
void test_many() {
  std::hive<int, A> h;
  std::vector<int> expected;
  for (int i = 0; i != 1000; ++i) {
    auto it = h.emplace(i);
    assert(*it == i);
    expected.push_back(i);
  }
  assert(h.size() == 1000);
  assert(h.capacity() >= 1000);
  std::vector<int> seen(h.begin(), h.end());
  std::sort(seen.begin(), seen.end());
  assert(seen == expected);
  check_invariant(h);
}

// Every element goes through the allocator's construct.
void test_allocator_construct() {
  test_allocator_statistics stats;
  {
    std::hive<int, test_allocator<int>> h{test_allocator<int>(&stats)};
    h.emplace(1);
    assert(stats.construct_count == 1);
    assert(stats.alloc_count > 0);
    h.emplace(2);
    assert(stats.construct_count == 2);
  }
  assert(stats.alloc_count == 0);
}

int main(int, char**) {
  test_returns_the_new_element<std::allocator<Pair>>();
  test_returns_the_new_element<min_allocator<Pair>>();
  test_move_only();
  test_constructs_exactly_one_object();
  test_argument_from_this<std::allocator<int>>();
  test_argument_from_this<min_allocator<int>>();
  test_many<std::allocator<int>>();
  test_many<min_allocator<int>>();
  test_allocator_construct();

  return 0;
}
