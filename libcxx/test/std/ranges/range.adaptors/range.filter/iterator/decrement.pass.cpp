//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iterator& operator--() requires bidirectional_range<V>;
// constexpr iterator operator--(int) requires bidirectional_range<V>;

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <iterator>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

struct EqualTo {
  int x;
  constexpr bool operator()(int y) const { return x == y; }
};

template <class T>
concept has_pre_decrement = requires (T t) { { --t }; };

template <class T>
concept has_post_decrement = requires (T t) { { t-- }; };

template <class Iterator>
using FilterIteratorFor = std::ranges::iterator_t<
  std::ranges::filter_view<minimal_view<Iterator, sentinel_wrapper<Iterator>>, EqualTo>
>;

template <class Iter, class Sent = sentinel_wrapper<Iter>>
constexpr void test() {
  using View = minimal_view<Iter, Sent>;
  using FilterView = std::ranges::filter_view<View, EqualTo>;
  using FilterIterator = std::ranges::iterator_t<FilterView>;

  auto make_filter_view = [](auto begin, auto end, auto pred) {
    View view{Iter(begin), Sent(Iter(end))};
    return FilterView(std::move(view), pred);
  };

  // Test with a single satisfied value
  {
    std::array<int, 5> array{0, 1, 2, 3, 4};
    FilterView view = make_filter_view(array.data(), array.data() + array.size(), EqualTo{1});
    FilterIterator it = std::ranges::next(view.begin(), view.end());
    assert(base(it.base()) == array.data() + array.size()); // test the test

    FilterIterator& result = --it;
    ASSERT_SAME_TYPE(FilterIterator&, decltype(--it));
    assert(&result == &it);
    assert(base(result.base()) == array.data() + 1);
  }

  // Test with more than one satisfied value
  {
    std::array<int, 6> array{0, 1, 2, 3, 1, 4};
    FilterView view = make_filter_view(array.data(), array.data() + array.size(), EqualTo{1});
    FilterIterator it = std::ranges::next(view.begin(), view.end());
    assert(base(it.base()) == array.data() + array.size()); // test the test

    FilterIterator& result = --it;
    assert(&result == &it);
    assert(base(result.base()) == array.data() + 4);

    --it;
    assert(base(it.base()) == array.data() + 1);
  }

  // Test going forward and then backward on the same iterator
  {
    std::array<int, 10> array{0, 1, 2, 3, 1, 1, 4, 5, 1, 6};
    FilterView view = make_filter_view(array.data(), array.data() + array.size(), EqualTo{1});
    FilterIterator it = view.begin();
    ++it;
    --it; assert(base(it.base()) == array.data() + 1);
    ++it; ++it;
    --it; assert(base(it.base()) == array.data() + 4);
    ++it; ++it;
    --it; assert(base(it.base()) == array.data() + 5);
    ++it; ++it;
    --it; assert(base(it.base()) == array.data() + 8);
  }

  // Test post-decrement
  {
    std::array<int, 6> array{0, 1, 2, 3, 1, 4};
    FilterView view = make_filter_view(array.data(), array.data() + array.size(), EqualTo{1});
    FilterIterator it = std::ranges::next(view.begin(), view.end());
    assert(base(it.base()) == array.data() + array.size()); // test the test

    FilterIterator result = it--;
    ASSERT_SAME_TYPE(FilterIterator, decltype(it--));
    assert(base(result.base()) == array.data() + array.size());
    assert(base(it.base()) == array.data() + 4);

    result = it--;
    assert(base(result.base()) == array.data() + 4);
    assert(base(it.base()) == array.data() + 1);
  }
}

constexpr bool tests() {
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();

  test<bidirectional_iterator<int const*>>();
  test<random_access_iterator<int const*>>();
  test<contiguous_iterator<int const*>>();
  test<int const*>();

  // Make sure `operator--` isn't provided for non bidirectional ranges
  {
    static_assert(!has_pre_decrement<FilterIteratorFor<cpp17_input_iterator<int*>>>);
    static_assert(!has_pre_decrement<FilterIteratorFor<cpp20_input_iterator<int*>>>);
    static_assert(!has_pre_decrement<FilterIteratorFor<forward_iterator<int*>>>);

    static_assert(!has_post_decrement<FilterIteratorFor<cpp17_input_iterator<int*>>>);
    static_assert(!has_post_decrement<FilterIteratorFor<cpp20_input_iterator<int*>>>);
    static_assert(!has_post_decrement<FilterIteratorFor<forward_iterator<int*>>>);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
