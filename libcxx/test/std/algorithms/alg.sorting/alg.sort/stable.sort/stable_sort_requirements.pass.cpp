//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <algorithm>

// template<class RandomAccessIterator, class Compare>
//   constexpr void                                                                       // constexpr since C++26
//     stable_sort(RandomAccessIterator first, RandomAccessIterator last, Compare comp);
//
// Preconditions: RandomAccessIterator meets the Cpp17ValueSwappable requirements, and the type of *first meets the
// Cpp17MoveConstructible and Cpp17MoveAssignable requirements.
//
// std::stable_sort can't rely on anything else: the elements don't have to be default constructible or copyable, the
// comparison only has to be callable with lvalues of the elements, and the iterators only have to be valid inside of
// the range.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include "operator_hijacker.h"
#include "stable_sort_helpers.h"
#include "test_macros.h"

template <std::size_t Padding>
void test_move_only_elements(int size) {
  typedef support::Probe<Padding> Probe;
  for (int pattern = 0; pattern != support::key_pattern_count; ++pattern) {
    std::vector<int> keys = support::make_keys(static_cast<support::KeyPattern>(pattern), size);
    {
      std::vector<Probe> probes = support::make_elements<Probe>(keys);
      std::stable_sort(probes.begin(), probes.end(), support::KeyLess());
      assert(support::keys_and_indices(probes) == support::stable_order(keys));
      assert(support::ProbeState::alive == size);
    }
    assert(support::ProbeState::alive == 0);
  }
}

template <std::size_t Padding, std::size_t N>
void test_move_only_elements(int max_dense_size, const int (&sizes)[N]) {
  for (int size = 0; size <= max_dense_size; ++size)
    test_move_only_elements<Padding>(size);
  for (std::size_t i = 0; i != N; ++i)
    test_move_only_elements<Padding>(sizes[i]);
}

// Forgets its key when it is moved from, so that the range ends up in the wrong order if the algorithm hands rvalues
// to a comparison that takes its arguments by value.
struct Forgetful {
  int key;
  int index;

  Forgetful(int k, int i) : key(k), index(i) {}
  Forgetful(const Forgetful&)            = default;
  Forgetful& operator=(const Forgetful&) = default;
  Forgetful(Forgetful&& other) : key(other.key), index(other.index) { other.key = -1; }
  Forgetful& operator=(Forgetful&& other) {
    key       = other.key;
    index     = other.index;
    other.key = -1;
    return *this;
  }
};

int key_of(const Forgetful& element) { return element.key; }
int index_of(const Forgetful& element) { return element.index; }

struct LessTakingValues {
  bool operator()(Forgetful a, Forgetful b) const { return a.key < b.key; }
};

void test_comparison_taking_values(int size) {
  std::vector<int> keys           = support::make_keys(support::random_keys, size);
  std::vector<Forgetful> elements = support::make_elements<Forgetful>(keys);
  std::stable_sort(elements.begin(), elements.end(), LessTakingValues());
  assert(support::keys_and_indices(elements) == support::stable_order(keys));
}

struct NonConstLess {
  template <class T>
  bool operator()(const T& a, const T& b) {
    return key_of(a) < key_of(b);
  }
};

void test_comparison_without_const_call_operator(int size) {
  typedef support::Element<0> Element;
  std::vector<int> keys         = support::make_keys(support::root_keys, size);
  std::vector<Element> elements = support::make_elements<Element>(keys);
  std::stable_sort(elements.begin(), elements.end(), NonConstLess());
  assert(support::keys_and_indices(elements) == support::stable_order(keys));
}

// A closure is neither default constructible nor assignable.
void test_comparison_that_is_only_copy_constructible(int size) {
  typedef support::Element<0> Element;
  std::vector<int> keys         = support::make_keys(support::quarter_keys, size);
  std::vector<Element> elements = support::make_elements<Element>(keys);
  int offset                    = 3;
  std::stable_sort(elements.begin(), elements.end(), [&offset](const Element& a, const Element& b) {
    return a.key + offset < b.key + offset;
  });
  assert(support::keys_and_indices(elements) == support::stable_order(keys));
}

template <class Element, class Difference>
void test_iterators_stay_inside_of_the_range(int size) {
  typedef support::CheckedIterator<Element, Difference> Iterator;
  for (int pattern = 0; pattern != support::key_pattern_count; ++pattern) {
    std::vector<int> keys         = support::make_keys(static_cast<support::KeyPattern>(pattern), size);
    std::vector<Element> elements = support::make_elements<Element>(keys);
    Element* first                = elements.data();
    Element* last                 = first + size;
    std::stable_sort(Iterator(first, first, last), Iterator(last, first, last), support::KeyLess());
    assert(support::keys_and_indices(elements) == support::stable_order(keys));
  }
}

template <class Element, class Difference, std::size_t N>
void test_iterators_stay_inside_of_the_range(const int (&sizes)[N]) {
  for (std::size_t i = 0; i != N; ++i)
    test_iterators_stay_inside_of_the_range<Element, Difference>(sizes[i]);
}

struct Hijacked : operator_hijacker {
  int key;
  int index;

  Hijacked(int k, int i) : key(k), index(i) {}
};

int key_of(const Hijacked& element) { return element.key; }
int index_of(const Hijacked& element) { return element.index; }

void test_elements_with_hijacked_operators(int size) {
  std::vector<int> keys          = support::make_keys(support::random_keys, size);
  std::vector<Hijacked> elements = support::make_elements<Hijacked>(keys);
  std::stable_sort(elements.begin(), elements.end(), support::KeyLess());
  assert(support::keys_and_indices(elements) == support::stable_order(keys));
}

int main(int, char**) {
  {
    const int sizes[] = {255, 256, 257, 511, 512, 513, 1000, 1025, 3001, 10007};
    test_move_only_elements<0>(150, sizes);
  }
  {
    const int sizes[] = {64, 65, 129, 257, 513};
    test_move_only_elements<1008>(40, sizes);
  }
  {
    const int sizes[] = {64, 65, 129, 257};
    test_move_only_elements<4096>(33, sizes);
  }

  {
    const int sizes[] = {0, 1, 2, 7, 8, 100, 1000, 3001};
    for (int size : sizes) {
      test_comparison_taking_values(size);
      test_comparison_without_const_call_operator(size);
      test_comparison_that_is_only_copy_constructible(size);
      test_elements_with_hijacked_operators(size);
    }
  }

  {
    const int sizes[] = {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 100, 257, 513, 1025, 3001};
    test_iterators_stay_inside_of_the_range<support::Element<0>, int>(sizes);
    test_iterators_stay_inside_of_the_range<support::Element<0>, long long>(sizes);
    test_iterators_stay_inside_of_the_range<support::Element<248>, int>(sizes);
    test_iterators_stay_inside_of_the_range<support::Element<248>, long long>(sizes);
  }
  {
    const int sizes[] = {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 64, 65, 129, 257};
    test_iterators_stay_inside_of_the_range<support::Element<1016>, int>(sizes);
    test_iterators_stay_inside_of_the_range<support::Element<4100>, int>(sizes);
    test_iterators_stay_inside_of_the_range<support::Element<4100>, long long>(sizes);
  }

  return 0;
}
