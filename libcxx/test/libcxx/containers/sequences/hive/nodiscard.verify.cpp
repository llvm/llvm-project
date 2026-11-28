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

// Check that <hive> functions are marked [[nodiscard]]

#include <hive>

template <class HiveT>
void test() {
  HiveT h;
  const std::hive_limits l = h.block_capacity_limits();

  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.get_allocator();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.begin();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.end();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.rbegin();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.rend();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.cbegin();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.cend();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.crbegin();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.crend();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.empty();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.size();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.capacity();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.block_capacity_limits();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.block_capacity_default_limits();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.block_capacity_hard_limits();
  // expected-warning@+1 2 {{ignoring return value of function declared with 'nodiscard' attribute}}
  h.is_within_hard_limits(l);
}

void instantiate() {
  test<std::hive<int> >();
  test<const std::hive<int> >();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hive<int>::block_capacity_default_limits();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hive<int>::block_capacity_hard_limits();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::hive<int>::is_within_hard_limits(std::hive_limits(1, 1));
}
