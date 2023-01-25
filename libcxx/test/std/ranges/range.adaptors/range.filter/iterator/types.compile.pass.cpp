//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::filter_view::<iterator>::difference_type
// std::filter_view::<iterator>::value_type
// std::filter_view::<iterator>::iterator_category
// std::filter_view::<iterator>::iterator_concept

#include <ranges>

#include <type_traits>
#include "test_iterators.h"
#include "../types.h"

template <typename T>
concept HasIteratorCategory = requires {
  typename T::iterator_category;
};

template <class Iterator>
using FilterViewFor = std::ranges::filter_view<
  minimal_view<Iterator, sentinel_wrapper<Iterator>>,
  AlwaysTrue
>;

template <class Iterator>
using FilterIteratorFor = std::ranges::iterator_t<FilterViewFor<Iterator>>;

struct ForwardIteratorWithInputCategory {
  using difference_type = int;
  using value_type = int;
  using iterator_category = std::input_iterator_tag;
  using iterator_concept = std::forward_iterator_tag;
  ForwardIteratorWithInputCategory();
  ForwardIteratorWithInputCategory& operator++();
  ForwardIteratorWithInputCategory operator++(int);
  int& operator*() const;
  friend bool operator==(ForwardIteratorWithInputCategory, ForwardIteratorWithInputCategory);
};
static_assert(std::forward_iterator<ForwardIteratorWithInputCategory>);

void f() {
  // Check that value_type is range_value_t and difference_type is range_difference_t
  {
    auto test = []<class Iterator> {
      using FilterView = FilterViewFor<Iterator>;
      using FilterIterator = FilterIteratorFor<Iterator>;
      static_assert(std::is_same_v<typename FilterIterator::value_type, std::ranges::range_value_t<FilterView>>);
      static_assert(std::is_same_v<typename FilterIterator::difference_type, std::ranges::range_difference_t<FilterView>>);
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
    static_assert(std::is_same_v<FilterIteratorFor<cpp17_input_iterator<int*>>::iterator_concept, std::input_iterator_tag>);
    static_assert(std::is_same_v<FilterIteratorFor<cpp20_input_iterator<int*>>::iterator_concept, std::input_iterator_tag>);
    static_assert(std::is_same_v<FilterIteratorFor<ForwardIteratorWithInputCategory>::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<FilterIteratorFor<forward_iterator<int*>>::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<FilterIteratorFor<bidirectional_iterator<int*>>::iterator_concept, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<FilterIteratorFor<random_access_iterator<int*>>::iterator_concept, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<FilterIteratorFor<contiguous_iterator<int*>>::iterator_concept, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<FilterIteratorFor<int*>::iterator_concept, std::bidirectional_iterator_tag>);
  }

  // Check iterator_category for various categories of ranges
  {
    static_assert(!HasIteratorCategory<FilterIteratorFor<cpp17_input_iterator<int*>>>);
    static_assert(!HasIteratorCategory<FilterIteratorFor<cpp20_input_iterator<int*>>>);
    static_assert(std::is_same_v<FilterIteratorFor<ForwardIteratorWithInputCategory>::iterator_category, std::input_iterator_tag>);
    static_assert(std::is_same_v<FilterIteratorFor<forward_iterator<int*>>::iterator_category, std::forward_iterator_tag>);
    static_assert(std::is_same_v<FilterIteratorFor<bidirectional_iterator<int*>>::iterator_category, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<FilterIteratorFor<random_access_iterator<int*>>::iterator_category, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<FilterIteratorFor<contiguous_iterator<int*>>::iterator_category, std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<FilterIteratorFor<int*>::iterator_category, std::bidirectional_iterator_tag>);
  }
}
