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

// UNSUPPORTED: no-exceptions

// __hive_storage::__emplace(args...): a throw from the constructor or the allocator leaves the storage unchanged.

#include <cassert>
#include <climits>
#include <hive>
#include <memory>

#include "test_allocator.h"
#include "test_macros.h"

template <class T, class A>
using Storage = std::__hive_storage<T, A>;

struct ThrowsOnce {
  static bool throw_next;
  int v;
  explicit ThrowsOnce(int x) : v(x) {
    if (throw_next) {
      throw_next = false;
      throw 42;
    }
  }
};
bool ThrowsOnce::throw_next = false;

template <class S>
void emplace_throws(S& s) {
  try {
    s.__emplace(0);
    assert(false);
  } catch (int) {
  }
}

// The constructor throws in a new block: the block is freed, nothing is counted.
void test_throw_in_new_block() {
  test_allocator_statistics stats;
  Storage<ThrowsOnce, test_allocator<ThrowsOnce>> s(std::hive_limits(3, 3), test_allocator<ThrowsOnce>(&stats));
  ThrowsOnce::throw_next = true;
  emplace_throws(s);
  assert(s.__size() == 0);
  assert(s.__capacity() == 0);
  assert(stats.alloc_count == 0);
  assert(s.__invariants());

  auto [b, i] = s.__emplace(1);
  assert(i == 0);
  assert(s.__size() == 1);
  assert(s.__capacity() == 3);
}

// The constructor throws in a free-list block: the block keeps its bits, sizes and place on the list.
void test_throw_in_free_list_block() {
  Storage<ThrowsOnce, std::allocator<ThrowsOnce>> s(std::hive_limits(3, 3), std::allocator<ThrowsOnce>{});
  auto [b, i]            = s.__emplace(1);
  ThrowsOnce::throw_next = true;
  emplace_throws(s);
  assert(s.__size() == 1);
  assert(s.__capacity() == 3);
  assert(b->__size_ == 1);
  assert(s.__invariants());

  auto [b2, i2] = s.__emplace(2);
  assert(b2 == b);
  assert(i2 == 1);
}

// The constructor throws in a reserved block: the block stays reserved and is taken next.
void test_throw_in_reserved_block() {
  Storage<ThrowsOnce, std::allocator<ThrowsOnce>> s(std::hive_limits(3, 3), std::allocator<ThrowsOnce>{});
  s.__emplace(1);
  s.__clear();
  ThrowsOnce::throw_next = true;
  emplace_throws(s);
  assert(s.__size() == 0);
  assert(s.__capacity() == 3);
  assert(s.__invariants());

  auto [b, i] = s.__emplace(2);
  assert(i == 0);
  assert(s.__capacity() == 3);
}

// The allocator throws on the slots (the first allocation of a block) or on the header (the second): nothing leaks.
void test_allocation_fails(int throw_after) {
  test_allocator_statistics stats;
  Storage<int, test_allocator<int>> s(std::hive_limits(3, 3), test_allocator<int>(&stats));
  stats.throw_after = throw_after;
  try {
    s.__emplace(0);
    assert(false);
  } catch (...) {
  }
  assert(s.__size() == 0);
  assert(s.__capacity() == 0);
  assert(stats.alloc_count == 0);
  assert(s.__invariants());

  stats.throw_after = INT_MAX;
  auto [b, i]       = s.__emplace(1);
  assert(i == 0);
  assert(s.__capacity() == 3);
  assert(stats.alloc_count == 2);
}

int main(int, char**) {
  test_throw_in_new_block();
  test_throw_in_free_list_block();
  test_throw_in_reserved_block();
  test_allocation_fails(0);
  test_allocation_fails(1);

  return 0;
}
