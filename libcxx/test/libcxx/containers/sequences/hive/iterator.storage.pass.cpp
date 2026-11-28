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

// __hive_storage::__begin() and __end(), and the iterator over blocks the storage built.

#include <cassert>
#include <concepts>
#include <hive>
#include <memory>

#include "min_allocator.h"
#include "test_macros.h"

template <class T, class A>
using Storage = std::__hive_storage<T, A>;

// Empty: begin and end are the value-initialized iterator.
template <class A>
void test_empty() {
  Storage<int, A> s{A()};
  const Storage<int, A>& cs                                                = s;
  std::same_as<typename Storage<int, A>::iterator> decltype(auto) b        = s.__begin();
  std::same_as<typename Storage<int, A>::iterator> decltype(auto) e        = s.__end();
  std::same_as<typename Storage<int, A>::const_iterator> decltype(auto) cb = cs.__begin();
  std::same_as<typename Storage<int, A>::const_iterator> decltype(auto) ce = cs.__end();
  assert(b == e);
  assert(cb == ce);
  assert((b == typename Storage<int, A>::iterator()));
  assert(b == cb);
}

// One element: begin is its slot, end is the block's capacity.
template <class A>
void test_one_element() {
  Storage<int, A> s(std::hive_limits(3, 3), A());
  auto [blk, i] = s.__emplace(42);
  auto it       = s.__begin();
  assert((it == std::__make_hive_iterator<typename Storage<int, A>::iterator>(blk, i)));
  assert(*it == 42);
  assert((s.__end() == std::__make_hive_iterator<typename Storage<int, A>::iterator>(blk, 3)));
  ++it;
  assert(it == s.__end());
  --it;
  assert(it == s.__begin());
}

// Seven elements in blocks of three: the walk reads them in order, forward and backward, through -> too.
template <class A>
void test_across_blocks() {
  Storage<int, A> s(std::hive_limits(3, 3), A());
  for (int n = 0; n != 7; ++n) {
    s.__emplace(n);
  }
  auto it = s.__begin();
  assert(*it == 0);
  ++it;
  assert(*it == 1);
  ++it;
  assert(*it == 2);
  ++it;
  assert(*it == 3);
  ++it;
  assert(*it == 4);
  ++it;
  assert(*it == 5);
  ++it;
  assert(*it == 6);
  ++it;
  assert(it == s.__end());
  --it;
  assert(*it == 6);
  --it;
  assert(*it == 5);
  --it;
  assert(*it == 4);
  --it;
  assert(*it == 3);
  --it;
  assert(*it == 2);
  --it;
  assert(*it == 1);
  --it;
  assert(*it == 0);
  assert(it == s.__begin());

  const Storage<int, A>& cs = s;
  auto cit                  = cs.__begin();
  assert(cit == s.__begin());
  assert(*cit == 0);
  assert(*cit.operator->() == 0);
  ++cit;
  assert(*cit == 1);
}

int main(int, char**) {
  test_empty<std::allocator<int>>();
  test_empty<min_allocator<int>>();
  test_one_element<std::allocator<int>>();
  test_one_element<min_allocator<int>>();
  test_across_blocks<std::allocator<int>>();
  test_across_blocks<min_allocator<int>>();

  return 0;
}
