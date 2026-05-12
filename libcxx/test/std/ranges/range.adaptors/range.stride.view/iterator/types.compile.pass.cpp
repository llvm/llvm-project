//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// stride_view::iterator::difference_type
// stride_view::iterator::value_type
// stride_view::iterator::iterator_concept
// stride_view::iterator::iterator_category

#include <ranges>
#include <type_traits>

#include "test_iterators.h"
#include "../types.h"

template <class T>
concept HasIteratorCategory = requires { typename T::iterator_category; };

template <class Iterator>
using StrideViewFor = std::ranges::stride_view<BasicTestView<Iterator, sentinel_wrapper<Iterator>>>;

template <class Iterator>
using StrideIteratorFor = std::ranges::iterator_t<StrideViewFor<Iterator>>;

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

// Non-simple view: forward when non-const, raw pointer when const.
// This exposes the bug where iterator_category is always derived from
// the non-const view's iterator instead of maybe-const<Const, V>.
struct DifferentCategoryView : std::ranges::view_base {
  forward_iterator<int*> begin();
  forward_iterator<int*> end();
  const int* begin() const;
  const int* end() const;
};

// Non-simple view: input-only when non-const, forward when const.
// Tests that iterator_category is absent for the non-const iterator
// but present for the const iterator.
struct ForwardWhenConstView : std::ranges::view_base {
  cpp17_input_iterator<int*> begin();
  sentinel_wrapper<cpp17_input_iterator<int*>> end();
  forward_iterator<const int*> begin() const;
  forward_iterator<const int*> end() const;
};

void f() {
  // Check that value_type is range_value_t and difference_type is range_difference_t
  {
    auto check = []<class Iterator> {
      using StrideView     = StrideViewFor<Iterator>;
      using StrideIterator = StrideIteratorFor<Iterator>;
      static_assert(std::is_same_v<typename StrideIterator::value_type, std::ranges::range_value_t<StrideView>>);
      static_assert(
          std::is_same_v<typename StrideIterator::difference_type, std::ranges::range_difference_t<StrideView>>);
    };
    check.operator()<cpp17_input_iterator<int*>>();
    check.operator()<forward_iterator<int*>>();
    check.operator()<bidirectional_iterator<int*>>();
    check.operator()<random_access_iterator<int*>>();
    check.operator()<contiguous_iterator<int*>>();
    check.operator()<int*>();
  }

  // Check iterator_concept for various categories of ranges
  {
    static_assert(
        std::is_same_v<StrideIteratorFor<cpp17_input_iterator<int*>>::iterator_concept, std::input_iterator_tag>);
    static_assert(
        std::is_same_v<StrideIteratorFor<forward_iterator<int*>>::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<StrideIteratorFor<ForwardIteratorWithInputCategory>::iterator_concept,
                                 std::forward_iterator_tag>);
    static_assert(std::is_same_v<StrideIteratorFor<bidirectional_iterator<int*>>::iterator_concept,
                                 std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<StrideIteratorFor<random_access_iterator<int*>>::iterator_concept,
                                 std::random_access_iterator_tag>);
    static_assert(std::is_same_v<StrideIteratorFor<contiguous_iterator<int*>>::iterator_concept,
                                 std::random_access_iterator_tag>);
    static_assert(std::is_same_v<StrideIteratorFor<int*>::iterator_concept, std::random_access_iterator_tag>);
  }

  // Check iterator_category for various categories of ranges
  {
    static_assert(!HasIteratorCategory<StrideIteratorFor<cpp17_input_iterator<int*>>>);
    static_assert(
        std::is_same_v<StrideIteratorFor<forward_iterator<int*>>::iterator_category, std::forward_iterator_tag>);
    static_assert(std::is_same_v<StrideIteratorFor<ForwardIteratorWithInputCategory>::iterator_category,
                                 std::input_iterator_tag>);
    static_assert(std::is_same_v<StrideIteratorFor<bidirectional_iterator<int*>>::iterator_category,
                                 std::bidirectional_iterator_tag>);
    static_assert(std::is_same_v<StrideIteratorFor<random_access_iterator<int*>>::iterator_category,
                                 std::random_access_iterator_tag>);
    static_assert(std::is_same_v<StrideIteratorFor<contiguous_iterator<int*>>::iterator_category,
                                 std::random_access_iterator_tag>);
    static_assert(std::is_same_v<StrideIteratorFor<int*>::iterator_category, std::random_access_iterator_tag>);
  }

  // Check that const vs non-const iterators get the correct iterator_category
  // when the view has different iterator categories for const and non-const.
  {
    using SV           = std::ranges::stride_view<DifferentCategoryView>;
    using NonConstIter = std::ranges::iterator_t<SV>;
    using ConstIter    = std::ranges::iterator_t<const SV>;

    static_assert(std::is_same_v<NonConstIter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::is_same_v<ConstIter::iterator_concept, std::random_access_iterator_tag>);

    static_assert(std::is_same_v<NonConstIter::iterator_category, std::forward_iterator_tag>);
    static_assert(std::is_same_v<ConstIter::iterator_category, std::random_access_iterator_tag>);
  }

  // Check that iterator_category presence/absence depends on the correct
  // const-qualified Base, not always the non-const view.
  {
    using SV           = std::ranges::stride_view<ForwardWhenConstView>;
    using NonConstIter = std::ranges::iterator_t<SV>;
    using ConstIter    = std::ranges::iterator_t<const SV>;

    static_assert(!HasIteratorCategory<NonConstIter>);
    static_assert(HasIteratorCategory<ConstIter>);
    static_assert(std::is_same_v<ConstIter::iterator_category, std::forward_iterator_tag>);
  }
}
