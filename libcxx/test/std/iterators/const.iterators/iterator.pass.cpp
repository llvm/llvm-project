//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::basic_const_iterator

#include <iterator>
#include <list>
#include <ranges>
#include <vector>
#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"

template <class T>
concept has_iterator_category = requires { typename T::iterator_category; };

template <class It>
constexpr bool check_category_and_concept() {
  using ConstIt = std::basic_const_iterator<It>;

  if constexpr (std::contiguous_iterator<It>) {
    ASSERT_SAME_TYPE(typename ConstIt::iterator_concept, std::contiguous_iterator_tag);
    static_assert(std::contiguous_iterator<ConstIt>);
  } else if constexpr (std::random_access_iterator<It>) {
    ASSERT_SAME_TYPE(typename ConstIt::iterator_concept, std::random_access_iterator_tag);
    static_assert(!std::contiguous_iterator<ConstIt>);
    static_assert(std::random_access_iterator<ConstIt>);
  } else if constexpr (std::bidirectional_iterator<It>) {
    ASSERT_SAME_TYPE(typename ConstIt::iterator_concept, std::bidirectional_iterator_tag);
    static_assert(!std::random_access_iterator<ConstIt>);
    static_assert(std::bidirectional_iterator<ConstIt>);
  } else if constexpr (std::forward_iterator<It>) {
    ASSERT_SAME_TYPE(typename ConstIt::iterator_concept, std::forward_iterator_tag);
    static_assert(!std::bidirectional_iterator<ConstIt>);
    static_assert(std::forward_iterator<ConstIt>);
  } else {
    ASSERT_SAME_TYPE(typename ConstIt::iterator_concept, std::input_iterator_tag);
    static_assert(!std::forward_iterator<ConstIt>);
    static_assert(std::input_iterator<ConstIt>);
  }

  if constexpr (std::forward_iterator<It>) {
    ASSERT_SAME_TYPE(typename ConstIt::iterator_category, typename std::iterator_traits<It>::iterator_category);
  } else {
    static_assert(!has_iterator_category<ConstIt>);
  }

  return true;
}

constexpr bool test_p2836r1() {
  auto f = [](std::vector<int>::const_iterator) {};

  auto v = std::vector<int>();
  {
    auto i1 = std::ranges::cbegin(v);
    f(i1);
  }

  auto t = v | std::views::take_while([](int const x) { return x < 100; });
  {
    auto i2 = std::ranges::cbegin(t);
    f(i2);
  }

  return true;
}

constexpr bool test_basic_operations() {
  struct S {
    int x;
  };
  S arr[10]                           = {};
  std::basic_const_iterator<S*> first = arr;
  std::basic_const_iterator<S*> last  = arr + 10;

  for (auto it = first; it != last; ++it) {
    (void)*it;
    (void)it->x;
    (void)iter_move(it);
  }
  static_assert(!std::is_invocable_v<decltype(std::ranges::iter_swap), decltype(first), decltype(first)>);

  assert(++first == arr + 1);
  assert(--first == arr + 0);
  assert(first++ == arr + 0);
  assert(first-- == arr + 1);

  assert(first + 3 == arr + 3);
  assert(last - 1 == arr + 9);

  first += 3;
  assert(first == arr + 3);
  first -= 2;
  assert(first == arr + 1);
  --first;

  assert(first < last);
  assert(last > first);
  assert(first <= last);
  assert(last >= first);

  assert(first < arr + 1);
  assert(arr + 1 > first);
  assert(first <= arr + 1);
  assert(arr + 1 >= first);

  assert((first <=> last) < 0);
  assert((first <=> arr + 1) < 0);
  assert((arr + 1 <=> first) > 0);

  return true;
}

int main() {
  types::for_each(types::cpp20_input_iterator_list<int*>{}, []<class It>() {
    using ConstIt = std::basic_const_iterator<It>;
    ASSERT_SAME_TYPE(typename ConstIt::value_type, int);
    static_assert(check_category_and_concept<It>());

    ASSERT_SAME_TYPE(std::iter_reference_t<ConstIt>, const int&);
    ASSERT_SAME_TYPE(std::iter_rvalue_reference_t<ConstIt>, const int&&);
  });

  test_p2836r1();
  static_assert(test_p2836r1());

  test_basic_operations();
  static_assert(test_basic_operations());

  return 0;
}
