//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <algorithm>

// template<class RandomAccessIterator>
//   constexpr void                                                                       // constexpr since C++26
//     stable_sort(RandomAccessIterator first, RandomAccessIterator last);
//
// template<class RandomAccessIterator, class Compare>
//   constexpr void                                                                       // constexpr since C++26
//     stable_sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp);
//
// Remarks: Stable.

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=200000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=200000000

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include "stable_sort_helpers.h"
#include "test_macros.h"

// The arrangement that std::stable_sort produces from elements whose keys are initially in the given order.
template <class Element>
TEST_CONSTEXPR_CXX26 std::vector<support::KeyAndIndex> sorted_with_comparator(const std::vector<int>& keys) {
  std::vector<Element> elements = support::make_elements<Element>(keys);
  std::stable_sort(elements.begin(), elements.end(), support::KeyLess());
  return support::keys_and_indices(elements);
}

template <class Element>
TEST_CONSTEXPR_CXX26 std::vector<support::KeyAndIndex> sorted_with_less_than(const std::vector<int>& keys) {
  std::vector<Element> elements = support::make_elements<Element>(keys);
  std::stable_sort(elements.begin(), elements.end());
  return support::keys_and_indices(elements);
}

template <class Element>
void test_comparator(int size) {
  for (int pattern = 0; pattern != support::key_pattern_count; ++pattern) {
    std::vector<int> keys = support::make_keys(static_cast<support::KeyPattern>(pattern), size);
    assert(sorted_with_comparator<Element>(keys) == support::stable_order(keys));
  }
}

template <class Element>
void test_less_than(int size) {
  for (int pattern = 0; pattern != support::key_pattern_count; ++pattern) {
    std::vector<int> keys = support::make_keys(static_cast<support::KeyPattern>(pattern), size);
    assert(sorted_with_less_than<Element>(keys) == support::stable_order(keys));
  }
}

template <class Element>
void test_comparator_up_to(int max_size) {
  for (int size = 0; size <= max_size; ++size)
    test_comparator<Element>(size);
}

template <class Element, std::size_t N>
void test_comparator_for_each(const int (&sizes)[N]) {
  for (std::size_t i = 0; i != N; ++i)
    test_comparator<Element>(sizes[i]);
}

template <class Element, std::size_t N>
void test_less_than_for_each(const int (&sizes)[N]) {
  for (std::size_t i = 0; i != N; ++i)
    test_less_than<Element>(sizes[i]);
}

// The sizes of the elements matter because an implementation may merge through a buffer whose capacity depends on
// them, down to no buffer at all for the largest elements.
void test_small_elements() {
  typedef support::Element<0> Element;
  const int sizes[] = {511, 512, 513, 1000, 1023, 1024, 1025, 2047, 2048, 2049, 4097, 10007, 65537, 100003};
  test_comparator_up_to<Element>(270);
  test_comparator_for_each<Element>(sizes);

  const int less_than_sizes[] = {0, 1, 2, 3, 7, 8, 9, 63, 64, 65, 128, 129, 257, 1000, 1025, 4097};
  test_less_than_for_each<Element>(less_than_sizes);
}

void test_32_byte_elements() {
  typedef support::Element<24> Element;
  const int sizes[] = {127, 128, 129, 255, 256, 257, 511, 512, 513, 1000, 2049, 5003, 40009};
  test_comparator_up_to<Element>(100);
  test_comparator_for_each<Element>(sizes);
  test_less_than_for_each<Element>(sizes);
}

void test_256_byte_elements() {
  typedef support::Element<248> Element;
  const int sizes[] = {100, 127, 128, 129, 255, 256, 257, 511, 513, 1000, 2049, 5003};
  test_comparator_up_to<Element>(70);
  test_comparator_for_each<Element>(sizes);
}

void test_1024_byte_elements() {
  typedef support::Element<1016> Element;
  const int sizes[] = {64, 65, 129, 257, 513, 1025};
  test_comparator_up_to<Element>(40);
  test_comparator_for_each<Element>(sizes);
}

void test_4096_byte_elements() {
  typedef support::Element<4088> Element;
  const int sizes[] = {64, 65, 129, 257};
  test_comparator_up_to<Element>(33);
  test_comparator_for_each<Element>(sizes);
}

void test_elements_larger_than_4096_bytes() {
  typedef support::Element<4100> Element;
  const int sizes[] = {64, 65, 129, 257};
  test_comparator_up_to<Element>(33);
  test_comparator_for_each<Element>(sizes);
}

// Block merge sorts work with blocks of about sqrt(N) elements, so that only a large range makes the blocks themselves
// large.
void test_large_range() {
  typedef support::Element<0> Element;
  const support::KeyPattern patterns[] = {
      support::random_keys, support::two_keys, support::root_keys, support::double_root_keys, support::descending_keys};
  for (support::KeyPattern pattern : patterns) {
    std::vector<int> keys = support::make_keys(pattern, 600011);
    assert(sorted_with_comparator<Element>(keys) == support::stable_order(keys));
  }
}

#if TEST_STD_VER >= 26
template <class Element>
constexpr bool sorts_stably_with_comparator(support::KeyPattern pattern, int size) {
  std::vector<int> keys = support::make_keys(pattern, size);
  return sorted_with_comparator<Element>(keys) == support::stable_order(keys);
}

template <class Element>
constexpr bool sorts_stably_with_less_than(support::KeyPattern pattern, int size) {
  std::vector<int> keys = support::make_keys(pattern, size);
  return sorted_with_less_than<Element>(keys) == support::stable_order(keys);
}

static_assert(sorts_stably_with_comparator<support::Element<0>>(support::random_keys, 7));
static_assert(sorts_stably_with_comparator<support::Element<0>>(support::random_keys, 300));
static_assert(sorts_stably_with_comparator<support::Element<0>>(support::random_keys, 1100));
static_assert(sorts_stably_with_comparator<support::Element<0>>(support::root_keys, 1100));
static_assert(sorts_stably_with_comparator<support::Element<0>>(support::descending_pair_keys, 1100));
static_assert(sorts_stably_with_less_than<support::Element<0>>(support::two_keys, 300));
static_assert(sorts_stably_with_comparator<support::Element<1016>>(support::random_keys, 70));
static_assert(sorts_stably_with_comparator<support::Element<1016>>(support::three_keys, 70));
static_assert(sorts_stably_with_comparator<support::Element<1016>>(support::descending_keys, 70));
#endif

int main(int, char**) {
  test_small_elements();
  test_32_byte_elements();
  test_256_byte_elements();
  test_1024_byte_elements();
  test_4096_byte_elements();
  test_elements_larger_than_4096_bytes();
  test_large_range();

  return 0;
}
