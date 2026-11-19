//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// The debug mode calls the comparison more often in order to validate it.
// UNSUPPORTED: libcpp-hardening-mode=debug

// <algorithm>

// std::stable_sort never copies the comparison it is given, and it calls it O(N log(N)) times whatever buffer it
// merges through, where the standard allows O(N log^2(N)) calls when no extra memory is available.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "stable_sort_helpers.h"
#include "test_macros.h"

struct Counters {
  long long calls;
  int copies;
};

struct CountingLess {
  Counters* counters;

  explicit CountingLess(Counters* c) : counters(c) {}
  CountingLess(const CountingLess& other) : counters(other.counters) { ++counters->copies; }
  CountingLess& operator=(const CountingLess&) = default;

  template <class T>
  bool operator()(const T& a, const T& b) const {
    ++counters->calls;
    return key_of(a) < key_of(b);
  }
};

template <class Element>
Counters sort_and_count(support::KeyPattern pattern, int size) {
  std::vector<Element> elements = support::make_elements<Element>(support::make_keys(pattern, size));
  Counters counters             = {0, 0};
  std::stable_sort(elements.begin(), elements.end(), CountingLess(&counters));
  return counters;
}

#if TEST_STD_VER >= 20
template <class Element>
Counters ranges_sort_and_count(support::KeyPattern pattern, int size) {
  std::vector<Element> elements = support::make_elements<Element>(support::make_keys(pattern, size));
  Counters counters             = {0, 0};
  std::ranges::stable_sort(elements, CountingLess(&counters));
  return counters;
}
#endif

template <class Element>
void test(int size) {
  for (int i = 0; i != support::key_pattern_count; ++i) {
    support::KeyPattern pattern = static_cast<support::KeyPattern>(i);
    Counters counters           = sort_and_count<Element>(pattern, size);
    assert(counters.copies == 0);
    assert(static_cast<double>(counters.calls) <= 3 * size * std::log2(size));
#if TEST_STD_VER >= 20
    assert(ranges_sort_and_count<Element>(pattern, size).copies == 0);
#endif
  }
}

int main(int, char**) {
  {
    const int sizes[] = {512, 513, 1000, 1025, 2049, 4097, 10007, 65537, 600011};
    for (int size : sizes)
      test<support::Element<0> >(size);
  }
  {
    const int sizes[] = {512, 513, 1000, 1025, 3001, 10007};
    for (int size : sizes)
      test<support::Probe<0> >(size);
  }
  {
    const int sizes[] = {512, 513, 1000, 2049, 5003};
    for (int size : sizes)
      test<support::Element<248> >(size);
  }
  {
    const int sizes[] = {512, 513, 1025};
    for (int size : sizes)
      test<support::Element<1016> >(size);
  }
  {
    const int sizes[] = {512, 513};
    for (int size : sizes)
      test<support::Element<4100> >(size);
  }

  return 0;
}
