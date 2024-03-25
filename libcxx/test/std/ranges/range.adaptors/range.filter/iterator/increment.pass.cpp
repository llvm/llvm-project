//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int) requires forward_range<V>;

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

struct EqualTo {
  int x;
  constexpr bool operator()(int y) const { return x == y; }
};

struct TrackingPred : TrackInitialization {
  using TrackInitialization::TrackInitialization;
  constexpr bool operator()(int i) const { return i == 1; }
};

template <class Iterator, bool IsForwardRange, bool IsConst>
constexpr void test() {
  using Sentinel = sentinel_wrapper<Iterator>;
  using View = minimal_view<Iterator, Sentinel>;
  using FilterView = std::ranges::filter_view<View, EqualTo>;
  using FilterIterator = std::ranges::iterator_t<FilterView>;

  auto make_filter_view = [](auto begin, auto end, auto pred) {
    View view{Iterator(begin), Sentinel(Iterator(end))};
    return FilterView(std::move(view), pred);
  };

  // Increment an iterator when it won't find another satisfied value after begin()
  {
    std::array<int, 5> array{0, 1, 2, 3, 4};
    FilterView view = make_filter_view(array.data(), array.data() + array.size(), EqualTo{1});

    FilterIterator it = view.begin();
    FilterIterator& result = ++it;
    ASSERT_SAME_TYPE(FilterIterator&, decltype(++it));
    assert(&result == &it);
    assert(base(result.base()) == array.data() + array.size());
  }

  // Increment the iterator and it finds another value after begin()
  {
    std::array<int, 5> array{99, 1, 99, 1, 99};
    FilterView view = make_filter_view(array.data(), array.data() + array.size(), EqualTo{1});

    FilterIterator it = view.begin();
    ++it;
    assert(base(it.base()) == array.data() + 3);
  }

  // Increment advances all the way to the end of the range
  {
    std::array<int, 5> array{99, 1, 99, 99, 1};
    FilterView view = make_filter_view(array.data(), array.data() + array.size(), EqualTo{1});

    FilterIterator it = view.begin();
    ++it;
    assert(base(it.base()) == array.data() + 4);
  }

  // Increment an iterator multiple times
  {
    std::array<int, 10> array{0, 1, 2, 3, 1, 1, 4, 5, 1, 6};
    FilterView view = make_filter_view(array.data(), array.data() + array.size(), EqualTo{1});

    FilterIterator it = view.begin();
          assert(base(it.base()) == array.data() + 1);
    ++it; assert(base(it.base()) == array.data() + 4);
    ++it; assert(base(it.base()) == array.data() + 5);
    ++it; assert(base(it.base()) == array.data() + 8);
    ++it; assert(base(it.base()) == array.data() + array.size());
  }

  // Test with a predicate that takes by non-const reference
  if constexpr (!IsConst) {
    std::array<int, 4> array{99, 1, 99, 1};
    View v{Iterator(array.data()), Sentinel(Iterator(array.data() + array.size()))};
    auto pred = [](int& x) { return x == 1; };
    auto view = std::ranges::filter_view(std::move(v), pred);
    auto it = view.begin();
    assert(base(it.base()) == array.data() + 1);
    ++it;
    assert(base(it.base()) == array.data() + 3);
  }

  // Make sure we do not make a copy of the predicate when we increment
  // (we should be passing it to ranges::find_if using std::ref)
  {
    bool moved = false, copied = false;
    std::array<int, 3> array{1, 1, 1};
    View v{Iterator(array.data()), Sentinel(Iterator(array.data() + array.size()))};
    auto view = std::ranges::filter_view(std::move(v), TrackingPred(&moved, &copied));
    moved = false;
    copied = false;
    auto it = view.begin();
    ++it;
    it++;
    assert(!moved);
    assert(!copied);
  }

  // Check post-increment for input ranges
  if constexpr (!IsForwardRange) {
    std::array<int, 10> array{0, 1, 2, 3, 1, 1, 4, 5, 1, 6};
    FilterView view = make_filter_view(array.data(), array.data() + array.size(), EqualTo{1});

    FilterIterator it = view.begin();
          assert(base(it.base()) == array.data() + 1);
    it++; assert(base(it.base()) == array.data() + 4);
    it++; assert(base(it.base()) == array.data() + 5);
    it++; assert(base(it.base()) == array.data() + 8);
    it++; assert(base(it.base()) == array.data() + array.size());
    static_assert(std::is_same_v<decltype(it++), void>);
  }

  // Check post-increment for forward ranges
  if constexpr (IsForwardRange) {
    std::array<int, 10> array{0, 1, 2, 3, 1, 1, 4, 5, 1, 6};
    FilterView view = make_filter_view(array.data(), array.data() + array.size(), EqualTo{1});

    FilterIterator it = view.begin();
    FilterIterator result = it++;
    ASSERT_SAME_TYPE(FilterIterator, decltype(it++));
    assert(base(result.base()) == array.data() + 1);
    assert(base(it.base()) == array.data() + 4);

    result = it++;
    assert(base(result.base()) == array.data() + 4);
    assert(base(it.base()) == array.data() + 5);

    result = it++;
    assert(base(result.base()) == array.data() + 5);
    assert(base(it.base()) == array.data() + 8);

    result = it++;
    assert(base(result.base()) == array.data() + 8);
    assert(base(it.base()) == array.data() + array.size());
  }
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>,   /* IsForwardRange */ false, /* IsConst */ false>();
  test<cpp20_input_iterator<int*>,   /* IsForwardRange */ false, /* IsConst */ false>();
  test<forward_iterator<int*>,       /* IsForwardRange */ true,  /* IsConst */ false>();
  test<bidirectional_iterator<int*>, /* IsForwardRange */ true,  /* IsConst */ false>();
  test<random_access_iterator<int*>, /* IsForwardRange */ true,  /* IsConst */ false>();
  test<contiguous_iterator<int*>,    /* IsForwardRange */ true,  /* IsConst */ false>();
  test<int*,                         /* IsForwardRange */ true,  /* IsConst */ false>();

  test<cpp17_input_iterator<int const*>,   /* IsForwardRange */ false, /* IsConst */ true>();
  test<cpp20_input_iterator<int const*>,   /* IsForwardRange */ false, /* IsConst */ true>();
  test<forward_iterator<int const*>,       /* IsForwardRange */ true,  /* IsConst */ true>();
  test<bidirectional_iterator<int const*>, /* IsForwardRange */ true,  /* IsConst */ true>();
  test<random_access_iterator<int const*>, /* IsForwardRange */ true,  /* IsConst */ true>();
  test<contiguous_iterator<int const*>,    /* IsForwardRange */ true,  /* IsConst */ true>();
  test<int const*,                         /* IsForwardRange */ true,  /* IsConst */ true>();
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
