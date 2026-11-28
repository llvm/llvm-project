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

// template<class RandomAccessIterator, class Compare>
//   constexpr void                                                                       // constexpr since C++26
//     stable_sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp);
//
// Complexity: Let N be last - first. If enough extra memory is available, N log(N) comparisons. Otherwise, at most
// N log^2(N) comparisons.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "stable_sort_helpers.h"
#include "test_macros.h"

struct CountingLess {
  long long* calls;

  template <class T>
  bool operator()(const T& a, const T& b) const {
    ++*calls;
    return key_of(a) < key_of(b);
  }
};

template <class Element>
long long comparisons(support::KeyPattern pattern, int size) {
  std::vector<Element> elements = support::make_elements<Element>(support::make_keys(pattern, size));
  long long calls               = 0;
  std::stable_sort(elements.begin(), elements.end(), CountingLess{&calls});
  return calls;
}

double max_comparisons(int size) {
  double log = std::log2(static_cast<double>(size));
  return size * log * log;
}

template <class Element>
void test(int size) {
  for (int pattern = 0; pattern != support::key_pattern_count; ++pattern)
    assert(static_cast<double>(comparisons<Element>(static_cast<support::KeyPattern>(pattern), size)) <=
           max_comparisons(size));
}

template <class Element, std::size_t N>
void test(int max_dense_size, const int (&sizes)[N]) {
  assert(comparisons<Element>(support::random_keys, 0) == 0);
  for (int size = 1; size <= max_dense_size; ++size)
    test<Element>(size);
  for (std::size_t i = 0; i != N; ++i)
    test<Element>(sizes[i]);
}

int main(int, char**) {
  {
    const int sizes[] = {511, 512, 513, 1000, 1024, 1025, 2049, 4097, 10007, 65537, 100003, 600011};
    test<support::Element<0> >(300, sizes);
  }
  {
    const int sizes[] = {511, 512, 513, 1000, 1025, 3001, 10007};
    test<support::Probe<0> >(300, sizes);
  }
  {
    const int sizes[] = {127, 128, 129, 255, 256, 257, 511, 513, 1000, 2049, 5003};
    test<support::Element<248> >(100, sizes);
  }
  {
    const int sizes[] = {100, 129, 257, 513, 1025};
    test<support::Element<1016> >(70, sizes);
  }
  {
    const int sizes[] = {64, 65, 100, 129, 257};
    test<support::Element<4100> >(40, sizes);
  }

  return 0;
}
