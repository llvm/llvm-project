//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <algorithm>

// std::__stable_sort merges through the buffer it is given whenever the runs or the blocks it merges fit into it, and
// falls back to merging through parts of the range otherwise. The capacity of the buffer that std::stable_sort
// provides only depends on the size of the elements, so this sorts with buffers of other capacities in order to go
// through every combination of these strategies with ranges of moderate sizes.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "stable_sort_helpers.h"
#include "test_macros.h"

typedef support::Probe<0> Probe;

void test(support::KeyPattern pattern, int size, std::ptrdiff_t capacity) {
  std::vector<int> keys = support::make_keys(pattern, size);
  {
    std::vector<Probe> probes = support::make_elements<Probe>(keys);
    std::allocator<Probe> allocator;
    Probe* buffer = allocator.allocate(static_cast<std::size_t>(capacity));
    // Makes a probe that is assigned to or read without having been constructed fail its assertions.
    std::memset(static_cast<void*>(buffer), 0xAB, static_cast<std::size_t>(capacity) * sizeof(Probe));
    support::KeyLess less;
    std::__stable_sort<std::_ClassicAlgPolicy, support::KeyLess&>(
        probes.begin(), probes.end(), less, probes.end() - probes.begin(), buffer, capacity);
    allocator.deallocate(buffer, static_cast<std::size_t>(capacity));
    assert(support::keys_and_indices(probes) == support::stable_order(keys));
    assert(support::ProbeState::alive == size);
  }
  assert(support::ProbeState::alive == 0);
}

void test(int size) {
  const std::ptrdiff_t capacities[] = {
      0,  1,  2,  3,  4,  5,   6,   7,   8,   9,   12,  15,   16,       17,       24,           31,   32,
      33, 48, 63, 64, 65, 100, 128, 200, 256, 400, 512, 1000, size / 4, size / 2, size / 2 + 1, size, size + 1};
  for (std::ptrdiff_t capacity : capacities)
    for (int pattern = 0; pattern != support::key_pattern_count; ++pattern)
      test(static_cast<support::KeyPattern>(pattern), size, capacity);
}

int main(int, char**) {
  const int sizes[] = {0,   1,   2,   3,   7,   8,   9,   15,  16,  17,  31,  32,   33,   63,   64,   65,   100,  127,
                       128, 129, 200, 255, 256, 257, 400, 511, 512, 513, 700, 1000, 1023, 1024, 1025, 1500, 2049, 3001};
  for (int size : sizes)
    test(size);

  return 0;
}
