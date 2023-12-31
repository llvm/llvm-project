//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// class enumerate_view

// class enumerate_view::iterator

// std::enumerate_view::<iterator>::difference_type;
// std::enumerate_view::<iterator>::value_type;
// std::enumerate_view::<iterator>::iterator_category;
// std::enumerate_view::<iterator>::iterator_concept;

#include <ranges>

#include <type_traits>
#include "test_iterators.h"
#include "../types.h"

template <typename T>
concept HasIteratorCategory = requires { typename T::iterator_category; };

template <class Iterator>
using EnumerateViewFor = std::ranges::enumerate_view< MinimalView<Iterator, sentinel_wrapper<Iterator>>>;

template <class Iterator>
using EnumerateIteratorFor = std::ranges::iterator_t<EnumerateViewFor<Iterator>>;

struct ForwardIteratorWithInputCategory {
  using difference_type   = int;
  using value_type        = int;
  using iterator_category = std::input_iterator_tag;
  using iterator_concept  = std::forward_iterator_tag;
  ForwardIteratorWithInputCategory();
  ForwardIteratorWithInputCategory& operator++();
  ForwardIteratorWithInputCategory operator++(int);
  int& operator*() const;
  friend bool operator==(ForwardIteratorWithInputCategory, ForwardIteratorWithInputCategory);
};
static_assert(std::forward_iterator<ForwardIteratorWithInputCategory>);

constexpr void test() {
  // Check that value_type is range_value_t and difference_type is range_difference_t
  {
    auto test = []<class Iterator> {
      using EnumerateView     = EnumerateViewFor<Iterator>;
      using EnumerateIterator = EnumerateIteratorFor<Iterator>;
      static_assert(std::is_same_v<typename EnumerateIterator::value_type, std::ranges::range_value_t<EnumerateView>>);
      static_assert(
          std::is_same_v<typename EnumerateIterator::difference_type, std::ranges::range_difference_t<EnumerateView>>);
    };
    test.operator()<cpp17_input_iterator<int*>>();
    test.operator()<cpp20_input_iterator<int*>>();
    test.operator()<forward_iterator<int*>>();
    test.operator()<bidirectional_iterator<int*>>();
    test.operator()<random_access_iterator<int*>>();
    test.operator()<contiguous_iterator<int*>>();
    test.operator()<int*>();
  }

  // Check iterator_concept for various categories of ranges
  {
    static_assert(
        std::is_same_v<EnumerateIteratorFor<cpp17_input_iterator<int*>>::iterator_concept, std::input_iterator_tag>);
    static_assert(
        std::is_same_v<EnumerateIteratorFor<cpp20_input_iterator<int*>>::iterator_concept, std::input_iterator_tag>);
    static_assert(std::is_same_v<EnumerateIteratorFor<ForwardIteratorWithInputCategory>::iterator_concept,
                                 std::forward_iterator_tag>);
    static_assert(
        std::is_same_v<EnumerateIteratorFor<forward_iterator<int*>>::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<EnumerateIteratorFor<bidirectional_iterator<int*>>::iterator_concept,
                                 std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<EnumerateIteratorFor<random_access_iterator<int*>>::iterator_concept,
                                 std::random_access_iterator_tag>);
    static_assert(std::is_same_v<EnumerateIteratorFor<contiguous_iterator<int*>>::iterator_concept,
                                 std::random_access_iterator_tag>);
    static_assert(std::is_same_v<EnumerateIteratorFor<int*>::iterator_concept, std::random_access_iterator_tag>);
  }

  // Check iterator_category for various categories of ranges
  {
    static_assert(HasIteratorCategory<EnumerateIteratorFor<cpp17_input_iterator<int*>>>);
    static_assert(HasIteratorCategory<EnumerateIteratorFor<cpp20_input_iterator<int*>>>);
    static_assert(std::is_same_v<EnumerateIteratorFor<ForwardIteratorWithInputCategory>::iterator_category,
                                 std::input_iterator_tag>);
    static_assert(
        std::is_same_v<EnumerateIteratorFor<forward_iterator<int*>>::iterator_category, std::input_iterator_tag>);
    static_assert(
        std::is_same_v<EnumerateIteratorFor<bidirectional_iterator<int*>>::iterator_category, std::input_iterator_tag>);
    static_assert(
        std::is_same_v<EnumerateIteratorFor<random_access_iterator<int*>>::iterator_category, std::input_iterator_tag>);
    static_assert(
        std::is_same_v<EnumerateIteratorFor<contiguous_iterator<int*>>::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<EnumerateIteratorFor<int*>::iterator_category, std::input_iterator_tag>);
  }
}
