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

// __hive_storage::__next_capacity(): the size, within the limits and rounded up to whole words, never above the
// maximum. Growth with the size is checked where the blocks are made, in storage.emplace.pass.cpp.

#include <cassert>
#include <cstddef>
#include <hive>
#include <memory>

#include "test_macros.h"

using Storage = std::__hive_storage<int, std::allocator<int>>;

// An empty storage asks for the minimum, rounded up to whole words within the maximum.
void test_empty(std::hive_limits limits, std::size_t expected) {
  Storage s(limits, std::allocator<int>());
  assert(s.__next_capacity() == expected);
}

int main(int, char**) {
  test_empty(std::hive<int>::block_capacity_default_limits(), 64);
  test_empty(std::hive_limits(1, 1), 1);
  test_empty(std::hive_limits(1, 4096), 64);
  test_empty(std::hive_limits(3, 3), 3);
  test_empty(std::hive_limits(65, 100), 100);
  test_empty(std::hive_limits(100, 200), 128);
  test_empty(std::hive_limits(4096, 4096), 4096);

  return 0;
}
