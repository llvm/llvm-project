//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive
// These modes assert instead, see assert.limits.pass.cpp.
// UNSUPPORTED: libcpp-hardening-mode=extensive, libcpp-hardening-mode=debug

// <hive>

// [hive.overview]: invalid limits are erroneous and their effects implementation-defined. libc++ clamps the maximum
// into the hard limits, then the minimum between the minimum hard limit and that maximum.

#include <cassert>
#include <cstddef>
#include <hive>
#include <limits>
#include <memory>

using H = std::hive<int>;

static_assert(H::block_capacity_hard_limits().min == 1);
static_assert(H::block_capacity_hard_limits().max == 4096);

void check(const H& h, std::size_t min, std::size_t max) {
  const std::hive_limits l = h.block_capacity_limits();
  assert(l.min == min);
  assert(l.max == max);
  assert(h.__invariants());
}

void test_minimum_below_hard_minimum() {
  check(H(std::hive_limits(0, 100)), 1, 100);
  check(H(std::hive_limits(0, 0)), 1, 1);
}

void test_maximum_above_hard_maximum() {
  check(H(std::hive_limits(10, 5000)), 10, 4096);
  check(H(std::hive_limits(5000, std::numeric_limits<std::size_t>::max())), 4096, 4096);
}

void test_minimum_above_maximum() {
  check(H(std::hive_limits(100, 50)), 50, 50);
  check(H(std::hive_limits(100, 50), std::allocator<int>()), 50, 50);
}

int main(int, char**) {
  test_minimum_below_hard_minimum();
  test_maximum_above_hard_maximum();
  test_minimum_above_maximum();

  return 0;
}
