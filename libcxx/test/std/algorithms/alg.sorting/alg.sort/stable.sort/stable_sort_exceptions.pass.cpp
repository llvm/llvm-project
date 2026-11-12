//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-exceptions

// <algorithm>

// template<class RandomAccessIterator, class Compare>
//   constexpr void                                                                       // constexpr since C++26
//     stable_sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp);
//
// When the comparison, a move or a swap of the elements throws, the exception leaves std::stable_sort and the range
// still holds one valid object per element: nothing has been destroyed, leaked or left unconstructed.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "stable_sort_helpers.h"
#include "test_macros.h"

struct ComparisonException {};

struct ThrowingLess {
  int* calls_before_throw;

  template <class T>
  bool operator()(const T& a, const T& b) const {
    if (*calls_before_throw == 0)
      throw ComparisonException();
    --*calls_before_throw;
    return key_of(a) < key_of(b);
  }
};

// Returns whether the comparison got to throw, which it doesn't once the sort needs fewer calls than that.
template <std::size_t Padding>
bool throws_from_comparison(support::KeyPattern pattern, int size, int calls_before_throw) {
  typedef support::Probe<Padding> Probe;
  std::vector<int> keys = support::make_keys(pattern, size);
  bool thrown           = false;
  {
    std::vector<Probe> probes = support::make_elements<Probe>(keys);
    try {
      std::stable_sort(probes.begin(), probes.end(), ThrowingLess{&calls_before_throw});
    } catch (const ComparisonException&) {
      thrown = true;
    }
    assert(support::ProbeState::alive == size);
    assert(thrown || support::keys_and_indices(probes) == support::stable_order(keys));
  }
  assert(support::ProbeState::alive == 0);
  return thrown;
}

template <std::size_t Padding>
bool throws_from_move(support::KeyPattern pattern, int size, int moves_before_throw) {
  typedef support::Probe<Padding> Probe;
  std::vector<int> keys = support::make_keys(pattern, size);
  bool thrown           = false;
  {
    std::vector<Probe> probes               = support::make_elements<Probe>(keys);
    support::ProbeState::moves_before_throw = moves_before_throw;
    try {
      std::stable_sort(probes.begin(), probes.end(), support::KeyLess());
    } catch (const support::ProbeException&) {
      thrown = true;
    }
    support::ProbeState::moves_before_throw = -1;
    assert(support::ProbeState::alive == size);
    assert(thrown || support::keys_and_indices(probes) == support::stable_order(keys));
  }
  assert(support::ProbeState::alive == 0);
  return thrown;
}

struct SwapException {};

int swaps_before_throw = -1;

template <std::size_t Padding>
struct SwappableProbe : support::Probe<Padding> {
  SwappableProbe(int key, int index) : support::Probe<Padding>(key, index) {}

  friend void swap(SwappableProbe& a, SwappableProbe& b) {
    assert(std::addressof(a) != std::addressof(b));
    if (swaps_before_throw == 0) {
      swaps_before_throw = -1;
      throw SwapException();
    }
    if (swaps_before_throw > 0)
      --swaps_before_throw;
    SwappableProbe temp(std::move(a));
    a = std::move(b);
    b = std::move(temp);
  }
};

template <std::size_t Padding>
bool throws_from_swap(support::KeyPattern pattern, int size, int swaps) {
  typedef SwappableProbe<Padding> Probe;
  std::vector<int> keys = support::make_keys(pattern, size);
  bool thrown           = false;
  {
    std::vector<Probe> probes = support::make_elements<Probe>(keys);
    swaps_before_throw        = swaps;
    try {
      std::stable_sort(probes.begin(), probes.end(), support::KeyLess());
    } catch (const SwapException&) {
      thrown = true;
    }
    swaps_before_throw = -1;
    assert(support::ProbeState::alive == size);
    assert(thrown || support::keys_and_indices(probes) == support::stable_order(keys));
  }
  assert(support::ProbeState::alive == 0);
  return thrown;
}

// Throwing from every single operation would take too long for the larger ranges, so the later an operation is, the
// more of its neighbours are skipped.
template <std::size_t Padding>
void test(int size) {
  for (int i = 0; i != support::key_pattern_count; ++i) {
    support::KeyPattern pattern = static_cast<support::KeyPattern>(i);
    for (int calls = 0; throws_from_comparison<Padding>(pattern, size, calls); calls += 1 + calls / 4) {
    }
    for (int moves = 0; throws_from_move<Padding>(pattern, size, moves); moves += 1 + moves / 4) {
    }
    for (int swaps = 0; throws_from_swap<Padding>(pattern, size, swaps); swaps += 1 + swaps / 4) {
    }
  }
}

int main(int, char**) {
  {
    const int sizes[] = {0, 1, 2, 3, 7, 8, 9, 16, 33, 100, 257, 600};
    for (int size : sizes)
      test<0>(size);
  }
  {
    const int sizes[] = {8, 16, 33, 70, 150};
    for (int size : sizes)
      test<1008>(size);
  }
  {
    const int sizes[] = {8, 16, 33, 70};
    for (int size : sizes)
      test<4096>(size);
  }

  return 0;
}
