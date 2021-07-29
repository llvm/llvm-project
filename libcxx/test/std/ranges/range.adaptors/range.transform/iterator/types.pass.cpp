//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// transform_view::<iterator>::difference_type
// transform_view::<iterator>::value_type
// transform_view::<iterator>::iterator_category
// transform_view::<iterator>::iterator_concept

#include <ranges>

#include "test_macros.h"
#include "../types.h"

template<class V, class F>
concept HasIterCategory = requires { typename std::ranges::transform_view<V, F>::iterator_category; };

constexpr bool test() {
  {
    // Member typedefs for contiguous iterator.
    static_assert(std::same_as<std::iterator_traits<int*>::iterator_concept, std::contiguous_iterator_tag>);
    static_assert(std::same_as<std::iterator_traits<int*>::iterator_category, std::random_access_iterator_tag>);

    using TView = std::ranges::transform_view<ContiguousView, IncrementRef>;
    using TIter = std::ranges::iterator_t<TView>;
    static_assert(std::same_as<typename TIter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::same_as<typename TIter::iterator_category, std::random_access_iterator_tag>);
    static_assert(std::same_as<typename TIter::value_type, int>);
    static_assert(std::same_as<typename TIter::difference_type, std::ptrdiff_t>);
  }
  {
    // Member typedefs for random access iterator.
    using TView = std::ranges::transform_view<RandomAccessView, IncrementRef>;
    using TIter = std::ranges::iterator_t<TView>;
    static_assert(std::same_as<typename TIter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::same_as<typename TIter::iterator_category, std::random_access_iterator_tag>);
    static_assert(std::same_as<typename TIter::value_type, int>);
    static_assert(std::same_as<typename TIter::difference_type, std::ptrdiff_t>);
  }
  {
    // Member typedefs for random access iterator/not-lvalue-ref.
    using TView = std::ranges::transform_view<RandomAccessView, Increment>;
    using TIter = std::ranges::iterator_t<TView>;
    static_assert(std::same_as<typename TIter::iterator_concept, std::random_access_iterator_tag>);
    static_assert(std::same_as<typename TIter::iterator_category, std::input_iterator_tag>); // Note: this is now input_iterator_tag.
    static_assert(std::same_as<typename TIter::value_type, int>);
    static_assert(std::same_as<typename TIter::difference_type, std::ptrdiff_t>);
  }
  {
    // Member typedefs for bidirectional iterator.
    using TView = std::ranges::transform_view<BidirectionalView, IncrementRef>;
    using TIter = std::ranges::iterator_t<TView>;
    static_assert(std::same_as<typename TIter::iterator_concept, std::bidirectional_iterator_tag>);
    static_assert(std::same_as<typename TIter::iterator_category, std::bidirectional_iterator_tag>);
    static_assert(std::same_as<typename TIter::value_type, int>);
    static_assert(std::same_as<typename TIter::difference_type, std::ptrdiff_t>);
  }
  {
    // Member typedefs for forward iterator.
    using TView = std::ranges::transform_view<ForwardView, IncrementRef>;
    using TIter = std::ranges::iterator_t<TView>;
    static_assert(std::same_as<typename TIter::iterator_concept, std::forward_iterator_tag>);
    static_assert(std::same_as<typename TIter::iterator_category, std::forward_iterator_tag>);
    static_assert(std::same_as<typename TIter::value_type, int>);
    static_assert(std::same_as<typename TIter::difference_type, std::ptrdiff_t>);
  }
  {
    // Member typedefs for input iterator.
    using TView = std::ranges::transform_view<InputView, IncrementRef>;
    using TIter = std::ranges::iterator_t<TView>;
    static_assert(std::same_as<typename TIter::iterator_concept, std::input_iterator_tag>);
    static_assert(!HasIterCategory<InputView, IncrementRef>);
    static_assert(std::same_as<typename TIter::value_type, int>);
    static_assert(std::same_as<typename TIter::difference_type, std::ptrdiff_t>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
