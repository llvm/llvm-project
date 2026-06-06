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
//                                       InputIterator2 first2, InputIterator2 last2,
//                                       Cmp comp)
//       -> decltype(comp(*b1, *b2));

#include <algorithm>
#include <array>
#include <cassert>
#include <compare>
#include <concepts>
#include <limits>
#include <vector>

#include "test_iterators.h"
#include "test_macros.h"

using std::array;

constexpr auto compare_last_digit_strong  = [](int a, int b) -> std::strong_ordering { return (a % 10) <=> (b % 10); };
constexpr auto compare_last_digit_weak    = [](int a, int b) -> std::weak_ordering { return (a % 10) <=> (b % 10); };
constexpr auto compare_last_digit_partial = [](int a, int b) -> std::partial_ordering {
  if (a == std::numeric_limits<int>::min() || b == std::numeric_limits<int>::min())
    return std::partial_ordering::unordered;
  return (a % 10) <=> (b % 10);
};
constexpr auto compare_int_result = [](int a, int b) -> int { return (a % 10) - (b % 10); };

struct StructWithoutCallOperator {};

template <class T>
concept has_lexicographical_compare =
    requires(int* ptr, T comp) { std::lexicographical_compare_three_way(ptr, ptr, ptr, ptr, comp); };

// `std::lexicographical_compare_three_way` accepts valid types
static_assert(has_lexicographical_compare<decltype(compare_last_digit_strong)>);
static_assert(has_lexicographical_compare<decltype(compare_last_digit_weak)>);
static_assert(has_lexicographical_compare<decltype(compare_last_digit_partial)>);
// `std::lexicographical_compare_three_way` rejects non-invocable comparators
static_assert(!has_lexicographical_compare<StructWithoutCallOperator>);
// `std::lexicographical_compare_three_way` accepts invalid comparators returning a wrong type.
// This will trigger a `static_assert` only when actually invoking `has_lexicographical_compare`.
static_assert(has_lexicographical_compare<decltype(compare_int_result)>);

template <typename Iter1, typename Iter2, typename C1, typename C2, typename Order, typename Comparator>
constexpr void test_lexicographical_compare(C1 a, C2 b, Comparator comp, Order expected) {
  std::same_as<Order> decltype(auto) result = std::lexicographical_compare_three_way(
      Iter1{a.data()}, Iter1{a.data() + a.size()}, Iter2{b.data()}, Iter2{b.data() + b.size()}, comp);
  assert(expected == result);
}

template <typename Iter1, typename Iter2>
constexpr void test_given_iterator_types() {
  auto cmp = compare_last_digit_strong;
  // Both inputs empty
  test_lexicographical_compare<Iter1, Iter2>(
      std::array<int, 0>{}, std::array<int, 0>{}, cmp, std::strong_ordering::equal);
  // Left input empty
  test_lexicographical_compare<Iter1, Iter2>(std::array<int, 0>{}, std::array{0, 1}, cmp, std::strong_ordering::less);
  // Right input empty
  test_lexicographical_compare<Iter1, Iter2>(
      std::array{0, 1}, std::array<int, 0>{}, cmp, std::strong_ordering::greater);

  // Identical arrays
  test_lexicographical_compare<Iter1, Iter2>(std::array{0, 1}, std::array{0, 1}, cmp, std::strong_ordering::equal);
  // "Less" on 2nd element
  test_lexicographical_compare<Iter1, Iter2>(std::array{0, 1}, std::array{0, 2}, cmp, std::strong_ordering::less);
  // "Greater" on 2nd element
  test_lexicographical_compare<Iter1, Iter2>(std::array{0, 2}, std::array{0, 1}, cmp, std::strong_ordering::greater);
  // "Greater" on 2nd element, but "less" on first entry
  test_lexicographical_compare<Iter1, Iter2>(std::array{0, 2}, std::array{1, 1}, cmp, std::strong_ordering::less);
  // Identical elements, but longer
  test_lexicographical_compare<Iter1, Iter2>(std::array{0, 1}, std::array{0, 1, 2}, cmp, std::strong_ordering::less);
  // Identical elements, but shorter
  test_lexicographical_compare<Iter1, Iter2>(std::array{0, 1, 2}, std::array{0, 1}, cmp, std::strong_ordering::greater);
  // Identical arrays, but only if we take the comparator
  // into account instead of using the default comparator
  test_lexicographical_compare<Iter1, Iter2>(std::array{10, 21}, std::array{10, 31}, cmp, std::strong_ordering::equal);
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
  test_lexicographical_compare<const int*, const int*>(
      std::array{0, 1}, std::array{10, 11}, compare_last_digit_weak, std::weak_ordering::equivalent);
  test_lexicographical_compare<const int*, const int*>(
      std::array{0, 1}, std::array{20, 11}, compare_last_digit_partial, std::partial_ordering::equivalent);

  // Check for all comparison categories with arrays of different sizes
  test_lexicographical_compare<const int*, const int*>(
      std::array{0}, std::array{0, 1}, compare_last_digit_strong, std::strong_ordering::less);
  test_lexicographical_compare<const int*, const int*>(
      std::array{0}, std::array{0, 1}, compare_last_digit_weak, std::weak_ordering::less);
  test_lexicographical_compare<const int*, const int*>(
      std::array{0}, std::array{0, 1}, compare_last_digit_partial, std::partial_ordering::less);

  // Check for a `partial_ordering::unordered` result
  test_lexicographical_compare<const int*, const int*>(
      std::array{std::numeric_limits<int>::min(), 1},
      std::array{0, 1, 2},
      compare_last_digit_partial,
      std::partial_ordering::unordered);
}

// Test for "Complexity: At most N applications of comp."
constexpr void test_comparator_invocation_count() {
  int compare_invocation_count     = 0;
  auto compare_last_digit_counting = [&](int a, int b) -> std::strong_ordering {
    ++compare_invocation_count;
    return (a % 10) <=> (b % 10);
  };
  // If one of the ranges is empty, the comparator must not be called at all
  compare_invocation_count = 0;
  test_lexicographical_compare<const int*, const int*>(
      std::array{0, 1, 2, 3}, std::array<int, 0>{}, compare_last_digit_counting, std::strong_ordering::greater);
  assert(compare_invocation_count == 0);
  // The comparator is invoked only `min(left.size(), right.size())` times
  test_lexicographical_compare<const int*, const int*>(
      std::array{0, 1, 2}, std::array{0, 1, 2, 3}, compare_last_digit_counting, std::strong_ordering::less);
#if defined(_LIBCPP_HARDENING_MODE) && _LIBCPP_HARDENING_MODE != _LIBCPP_HARDENING_MODE_DEBUG
  assert(compare_invocation_count <= 3);
#else
  assert(compare_invocation_count <= 6);
#endif
}

// Check that it works with proxy iterators
constexpr void test_proxy_iterators() {
    std::vector<bool> vec(10, true);
    auto result = std::lexicographical_compare_three_way(vec.begin(), vec.end(), vec.begin(), vec.end(), compare_last_digit_strong);
    assert(result == std::strong_ordering::equal);
}

constexpr bool test() {
  test_iterator_types();
  test_comparison_categories();
  test_comparator_invocation_count();
  test_proxy_iterators();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
