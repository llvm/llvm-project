//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<class InputIterator1, class InputIterator2, class Cmp>
//     constexpr auto
//     lexicographical_compare_three_way(InputIterator1 first1, InputIterator1 last1,
//                                       InputIterator2 first2, InputIterator2 last2)

#include <array>
#include <algorithm>
#include <cassert>
#include <compare>
#include <concepts>

#include "test_macros.h"
#include "test_comparisons.h"
#include "test_iterators.h"

template <typename Iter1, typename Iter2, typename C1, typename C2, typename Order>
constexpr void test_lexicographical_compare(C1 a, C2 b, Order expected) {
  std::same_as<Order> decltype(auto) result =
      std::lexicographical_compare_three_way(Iter1{a.begin()}, Iter1{a.end()}, Iter2{b.begin()}, Iter2{b.end()});
  assert(expected == result);
}

template <typename Iter1, typename Iter2>
constexpr void test_given_iterator_types() {
  // Both inputs empty
  test_lexicographical_compare<Iter1, Iter2>(std::array<int, 0>{}, std::array<int, 0>{}, std::strong_ordering::equal);
  // Left input empty
  test_lexicographical_compare<Iter1, Iter2>(std::array<int, 0>{}, std::array{0, 1}, std::strong_ordering::less);
  // Right input empty
  test_lexicographical_compare<Iter1, Iter2>(std::array{0, 1}, std::array<int, 0>{}, std::strong_ordering::greater);

  // Identical arrays
  test_lexicographical_compare<Iter1, Iter2>(std::array{0, 1}, std::array{0, 1}, std::strong_ordering::equal);
  // "Less" on 2nd element
  test_lexicographical_compare<Iter1, Iter2>(std::array{0, 1}, std::array{0, 2}, std::strong_ordering::less);
  // "Greater" on 2nd element
  test_lexicographical_compare<Iter1, Iter2>(std::array{0, 2}, std::array{0, 1}, std::strong_ordering::greater);
  // "Greater" on 2nd element, but "less" on first entry
  test_lexicographical_compare<Iter1, Iter2>(std::array{0, 2}, std::array{1, 1}, std::strong_ordering::less);
  // Identical elements, but longer
  test_lexicographical_compare<Iter1, Iter2>(std::array{0, 1}, std::array{0, 1, 2}, std::strong_ordering::less);
  // Identical elements, but shorter
  test_lexicographical_compare<Iter1, Iter2>(std::array{0, 1, 2}, std::array{0, 1}, std::strong_ordering::greater);
}

template <typename Iter1>
constexpr void test_iterator_types1() {
  test_given_iterator_types<Iter1, int*>();
  test_given_iterator_types<Iter1, const int*>();
  test_given_iterator_types<Iter1, cpp17_input_iterator<const int*>>();
  test_given_iterator_types<Iter1, forward_iterator<const int*>>();
  test_given_iterator_types<Iter1, bidirectional_iterator<const int*>>();
  test_given_iterator_types<Iter1, random_access_iterator<const int*>>();
  test_given_iterator_types<Iter1, contiguous_iterator<const int*>>();
}

constexpr void test_iterator_types() {
  // Exhaustively test all combinations of `int*`, `const int*`, `cpp17_input_iterator`,
  // `forward_iterator`, `bidirectional_iterator`, `random_access_iterator`,
  // `contiguous_iterator`.
  //
  // `lexicographical_compare_three_way` has a fast path which triggers if both
  // iterators are random access iterators.

  test_iterator_types1<int*>();
  test_iterator_types1<const int*>();
  test_iterator_types1<cpp17_input_iterator<const int*>>();
  test_iterator_types1<forward_iterator<const int*>>();
  test_iterator_types1<bidirectional_iterator<const int*>>();
  test_iterator_types1<random_access_iterator<const int*>>();
  test_iterator_types1<contiguous_iterator<const int*>>();
}

// Check for other comparison categories
constexpr void test_comparison_categories() {
  // Check all comparison categories for inputs with a difference in the contained elements
  test_lexicographical_compare<const StrongOrder*, const StrongOrder*>(
      std::array<StrongOrder, 2>{0, 1}, std::array<StrongOrder, 2>{1, 1}, std::strong_ordering::less);
  test_lexicographical_compare<const WeakOrder*, const WeakOrder*>(
      std::array<WeakOrder, 2>{0, 1}, std::array<WeakOrder, 2>{1, 1}, std::weak_ordering::less);
  test_lexicographical_compare<const PartialOrder*, const PartialOrder*>(
      std::array<PartialOrder, 2>{0, 1}, std::array<PartialOrder, 2>{1, 1}, std::partial_ordering::less);

  // Check comparison categories with arrays of different sizes
  test_lexicographical_compare<const StrongOrder*, const StrongOrder*>(
      std::array<StrongOrder, 2>{0, 1}, std::array<StrongOrder, 3>{0, 1, 2}, std::strong_ordering::less);
  test_lexicographical_compare<const WeakOrder*, const WeakOrder*>(
      std::array<WeakOrder, 2>{0, 1}, std::array<WeakOrder, 3>{0, 1, 2}, std::weak_ordering::less);
  test_lexicographical_compare<const PartialOrder*, const PartialOrder*>(
      std::array<PartialOrder, 2>{0, 1}, std::array<PartialOrder, 3>{0, 1, 2}, std::partial_ordering::less);

  // Check for a `partial_ordering::unordered` result
  test_lexicographical_compare<const PartialOrder*, const PartialOrder*>(
      std::array<PartialOrder, 2>{std::numeric_limits<int>::min(), 1},
      std::array<PartialOrder, 3>{0, 1, 2},
      std::partial_ordering::unordered);
}

constexpr bool test() {
  test_iterator_types();
  test_comparison_categories();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
