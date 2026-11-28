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

// The growth of active block capacities, as documented in ImplementationDefinedBehavior.md: a new block gets the
// size at the time, clamped to the limits and rounded up to a multiple of 64 within the maximum.

#include <cassert>
#include <cstddef>
#include <hive>
#include <utility>
#include <vector>

#include "test_macros.h"

// Inserts elements one by one and checks the capacity after the counts given.
void check_growth(std::hive_limits limits, const std::vector<std::pair<std::size_t, std::size_t>>& after) {
  std::hive<int> h(limits);
  std::size_t n = 0;
  for (const auto& [count, capacity] : after) {
    while (n != count) {
      h.insert(0);
      ++n;
    }
    assert(h.size() == count);
    assert(h.capacity() == capacity);
  }
}

int main(int, char**) {
  // Default limits for int: {64, 4096}.
  check_growth(
      std::hive<int>::block_capacity_default_limits(),
      {{1, 64},
       {64, 64},
       {65, 128},
       {129, 256},
       {257, 512},
       {513, 1024},
       {1025, 2048},
       {2049, 4096},
       {4097, 8192},
       {8193, 12288}});
  check_growth(std::hive_limits(3, 3), {{1, 3}, {3, 3}, {4, 6}, {7, 9}});
  check_growth(std::hive_limits(65, 100), {{1, 100}, {101, 200}});
  check_growth(std::hive_limits(100, 200), {{1, 128}, {129, 256}, {257, 456}});

  return 0;
}
