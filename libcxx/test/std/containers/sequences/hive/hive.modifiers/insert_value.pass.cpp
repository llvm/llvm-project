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

// iterator insert(const T& x);
// iterator insert(T&& x);
// iterator insert(const_iterator hint, const T& x);
// iterator insert(const_iterator hint, T&& x);

#include <algorithm>
#include <cassert>
#include <concepts>
#include <hive>
#include <memory>

#include "../helpers.h"
#include "MoveOnly.h"
#include "min_allocator.h"
#include "test_macros.h"

struct Tracked {
  static int copies;
  static int moves;
  int v;
  explicit Tracked(int x) : v(x) {}
  Tracked(const Tracked& o) : v(o.v) { ++copies; }
  Tracked(Tracked&& o) : v(o.v) { ++moves; }
  static void reset() {
    copies = 0;
    moves  = 0;
  }
};
int Tracked::copies = 0;
int Tracked::moves  = 0;

// An lvalue is copied once, an rvalue moved once, with or without the hint; the result points to the element.
template <class A>
void test_copies_and_moves() {
  using H = std::hive<Tracked, A>;
  H h;
  const Tracked x(1);

  Tracked::reset();
  std::same_as<typename H::iterator> decltype(auto) it = h.insert(x);
  assert(it->v == 1);
  assert(Tracked::copies == 1);
  assert(Tracked::moves == 0);
  assert(h.size() == 1);
  check_invariant(h);

  Tracked::reset();
  std::same_as<typename H::iterator> decltype(auto) it2 = h.insert(Tracked(2));
  assert(it2->v == 2);
  assert(Tracked::copies == 0);
  assert(Tracked::moves == 1);
  assert(h.size() == 2);
  check_invariant(h);

  Tracked::reset();
  std::same_as<typename H::iterator> decltype(auto) it3 = h.insert(h.cbegin(), x);
  assert(it3->v == 1);
  assert(Tracked::copies == 1);
  assert(Tracked::moves == 0);
  assert(h.size() == 3);
  check_invariant(h);

  Tracked::reset();
  std::same_as<typename H::iterator> decltype(auto) it4 = h.insert(h.end(), Tracked(4));
  assert(it4->v == 4);
  assert(Tracked::copies == 0);
  assert(Tracked::moves == 1);
  assert(h.size() == 4);
  check_invariant(h);

  assert(it->v == 1);
  assert(it2->v == 2);
  assert(it3->v == 1);
}

// A move-only type takes the rvalue overloads.
void test_move_only() {
  std::hive<MoveOnly> h;
  auto it = h.insert(MoveOnly(5));
  assert(it->get() == 5);
  auto it2 = h.insert(h.begin(), MoveOnly(6));
  assert(it2->get() == 6);
  assert(h.size() == 2);
  check_invariant(h);
}

// Inserting an element of the hive itself.
template <class A>
void test_insert_from_this() {
  std::hive<int, A> h;
  h.insert(3);
  h.insert(*h.begin());
  h.insert(h.cbegin(), *h.begin());
  assert(h.size() == 3);
  assert(std::count(h.begin(), h.end(), 3) == 3);
  check_invariant(h);
}

int main(int, char**) {
  test_copies_and_moves<std::allocator<Tracked>>();
  test_copies_and_moves<min_allocator<Tracked>>();
  test_move_only();
  test_insert_from_this<std::allocator<int>>();
  test_insert_from_this<min_allocator<int>>();

  return 0;
}
