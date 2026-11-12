//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <algorithm>

// std::stable_sort and std::ranges::stable_sort sort in place: they never allocate memory.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include "count_new.h"
#include "stable_sort_helpers.h"
#include "test_macros.h"

template <class Element>
void test(support::KeyPattern pattern, int size) {
  std::vector<int> keys         = support::make_keys(pattern, size);
  std::vector<Element> elements = support::make_elements<Element>(keys);
  {
    DisableAllocationGuard guard;
    std::stable_sort(elements.begin(), elements.end(), support::KeyLess());
  }
  assert(support::keys_and_indices(elements) == support::stable_order(keys));
}

#if TEST_STD_VER >= 20
template <class Element>
void test_ranges(support::KeyPattern pattern, int size) {
  std::vector<int> keys         = support::make_keys(pattern, size);
  std::vector<Element> elements = support::make_elements<Element>(keys);
  {
    DisableAllocationGuard guard;
    std::ranges::stable_sort(elements, support::KeyLess());
  }
  assert(support::keys_and_indices(elements) == support::stable_order(keys));
}
#endif

// The sizes are the ones around which the radix sort for arithmetic types, which needs a buffer as large as the range,
// could be used.
template <class Arithmetic>
void test_arithmetic(int size) {
  std::vector<int> keys = support::make_keys(support::random_keys, size);
  std::vector<Arithmetic> values;
  for (int key : keys)
    values.push_back(static_cast<Arithmetic>(key % 100));
  std::vector<Arithmetic> expected = values;
  std::sort(expected.begin(), expected.end());
  {
    DisableAllocationGuard guard;
    std::stable_sort(values.begin(), values.end());
  }
  assert(values == expected);
}

template <class Element>
void test(int size) {
  for (int pattern = 0; pattern != support::key_pattern_count; ++pattern) {
    test<Element>(static_cast<support::KeyPattern>(pattern), size);
#if TEST_STD_VER >= 20
    test_ranges<Element>(static_cast<support::KeyPattern>(pattern), size);
#endif
  }
}

int main(int, char**) {
  {
    const int sizes[] = {0, 1, 7, 8, 100, 129, 1000, 1025, 4097, 100003};
    for (int size : sizes)
      test<support::Element<0> >(size);
  }
  {
    const int sizes[] = {0, 1, 7, 8, 100, 129, 1000, 5003};
    for (int size : sizes)
      test<support::Element<248> >(size);
  }
  {
    const int sizes[] = {0, 1, 7, 8, 100, 257};
    for (int size : sizes)
      test<support::Element<4100> >(size);
  }
  {
    const int sizes[] = {0, 1, 64, 65, 255, 256, 257, 512, 513, 1023, 1024, 1025, 65536, 65537, 100003};
    for (int size : sizes) {
      test_arithmetic<signed char>(size);
      test_arithmetic<int>(size);
      test_arithmetic<long long>(size);
      test_arithmetic<double>(size);
    }
  }

  return 0;
}
