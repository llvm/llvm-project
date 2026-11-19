//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// The debug mode notices that the comparisons aren't strict weak orders and terminates the program.
// UNSUPPORTED: libcpp-hardening-mode=debug

// <algorithm>

// Sorting with a comparison that isn't a strict weak order has undefined behavior. Even then, std::stable_sort
// terminates, stays inside of the range it was given, and leaves a permutation of the elements in it.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include "stable_sort_helpers.h"
#include "test_macros.h"

enum Rule {
  always_true,
  less_or_equal,
  greater_or_equal,
  not_equal,
  cyclic,
  alternating,
  scrambled,
  less_than_with_scrambled_ties,
  rule_count
};

struct InvalidComparison {
  Rule rule;
  long long* calls;

  template <class T>
  bool operator()(const T& a, const T& b) const {
    ++*calls;
    switch (rule) {
    case always_true:
      return true;
    case less_or_equal:
      return key_of(a) <= key_of(b);
    case greater_or_equal:
      return key_of(a) >= key_of(b);
    case not_equal:
      return key_of(a) != key_of(b);
    case cyclic:
      return (key_of(a) + 1) % 3 == key_of(b) % 3;
    case alternating:
      return *calls % 2 == 0;
    case scrambled:
      return (key_of(a) * 31LL + key_of(b) * 17LL + *calls * 7) % 5 < 2;
    case less_than_with_scrambled_ties:
      return key_of(a) < key_of(b) || (key_of(a) == key_of(b) && *calls % 3 == 0);
    case rule_count:
      break;
    }
    return false;
  }
};

template <class Range>
std::vector<int> sorted_indices(const Range& range) {
  std::vector<int> indices;
  for (auto it = range.begin(); it != range.end(); ++it)
    indices.push_back(index_of(*it));
  std::sort(indices.begin(), indices.end());
  return indices;
}

std::vector<int> all_indices(int size) {
  std::vector<int> indices;
  for (int i = 0; i != size; ++i)
    indices.push_back(i);
  return indices;
}

// With such comparisons the merges can degrade to moving every element past every other one, so the limit is
// quadratic. A loop that doesn't make progress still reaches it quickly.
long long max_operations(int size) { return 1000000 + 200LL * size * size; }

template <class Element>
void test(Rule rule, support::KeyPattern pattern, int size) {
  typedef support::CheckedIterator<Element> Iterator;
  std::vector<Element> elements = support::make_elements<Element>(support::make_keys(pattern, size));
  Element* first                = elements.data();
  Element* last                 = first + size;
  long long calls               = 0;

  support::CheckedIteratorState::operations_left = max_operations(size);
  std::stable_sort(Iterator(first, first, last), Iterator(last, first, last), InvalidComparison{rule, &calls});

  assert(calls <= max_operations(size));
  assert(sorted_indices(elements) == all_indices(size));
}

template <class Element>
void test(int size) {
  const support::KeyPattern patterns[] = {
      support::ascending_keys,
      support::descending_keys,
      support::equal_keys,
      support::random_keys,
      support::three_keys,
      support::root_keys,
      support::sawtooth_keys,
      support::distinct_window_keys};
  for (int rule = 0; rule != rule_count; ++rule)
    for (support::KeyPattern pattern : patterns)
      test<Element>(static_cast<Rule>(rule), pattern, size);
}

int main(int, char**) {
  {
    const int sizes[] = {100, 129, 257, 513, 1000, 1025, 2049, 3001};
    for (int size = 0; size <= 70; ++size)
      test<support::Element<0> >(size);
    for (int size : sizes)
      test<support::Element<0> >(size);
  }
  {
    const int sizes[] = {100, 129, 257, 513, 1025};
    for (int size = 0; size <= 70; ++size)
      test<support::Element<248> >(size);
    for (int size : sizes)
      test<support::Element<248> >(size);
  }
  {
    const int sizes[] = {64, 65, 100, 129, 257};
    for (int size = 0; size <= 40; ++size)
      test<support::Element<4100> >(size);
    for (int size : sizes)
      test<support::Element<4100> >(size);
  }

  return 0;
}
