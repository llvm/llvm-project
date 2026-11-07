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

// template<class... Args> iterator emplace_hint(const_iterator hint, Args&&... args);

#include <algorithm>
#include <cassert>
#include <concepts>
#include <hive>
#include <memory>

#include "../helpers.h"
#include "Counter.h"
#include "min_allocator.h"
#include "test_macros.h"

struct Pair {
  int a;
  int b;
  Pair(int x, int y) : a(x), b(y) {}
};

// The hint is ignored: any const_iterator value, and the result points to the new element.
template <class A>
void test_hint_is_ignored() {
  using H = std::hive<Pair, A>;
  H h;
  std::same_as<typename H::iterator> decltype(auto) it = h.emplace_hint(h.begin(), 1, 2);
  assert(it->a == 1);
  assert(it->b == 2);
  assert(h.size() == 1);
  assert(it == h.begin());
  check_invariant(h);

  std::same_as<typename H::iterator> decltype(auto) it2 = h.emplace_hint(h.cend(), 3, 4);
  assert(it2->a == 3);
  assert(it2 != it);
  assert(h.size() == 2);
  check_invariant(h);

  std::same_as<typename H::iterator> decltype(auto) it3 = h.emplace_hint(typename H::const_iterator(), 5, 6);
  assert(it3->a == 5);
  assert(h.size() == 3);
  assert(std::count_if(h.begin(), h.end(), [](const Pair& p) { return p.a == 5; }) == 1);
  check_invariant(h);
}

// Exactly one object is constructed per call.
void test_constructs_exactly_one_object() {
  {
    std::hive<Counter<int>> h;
    h.emplace_hint(h.end(), 1);
    assert(Counter_base::gConstructed == 1);
    h.emplace_hint(h.begin(), Counter<int>(2));
    assert(Counter_base::gConstructed == 2);
  }
  assert(Counter_base::gConstructed == 0);
}

int main(int, char**) {
  test_hint_is_ignored<std::allocator<Pair>>();
  test_hint_is_ignored<min_allocator<Pair>>();
  test_constructs_exactly_one_object();

  return 0;
}
