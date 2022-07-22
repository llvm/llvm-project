//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// <algorithm>

// template<forward_iterator I, sentinel_for<I> S, class T, class Proj = identity,
//          indirect_strict_weak_order<const T*, projected<I, Proj>> Comp = ranges::less>
//   constexpr subrange<I>
//     equal_range(I first, S last, const T& value, Comp comp = {}, Proj proj = {});                // Since C++20
//
// template<forward_range R, class T, class Proj = identity,
//          indirect_strict_weak_order<const T*, projected<iterator_t<R>, Proj>> Comp =
//            ranges::less>
//   constexpr borrowed_subrange_t<R>
//     equal_range(R&& r, const T& value, Comp comp = {}, Proj proj = {});                          // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"
#include "../../sortable_helpers.h"

struct Foo {};

template <class Iter, class Sent, class T, class Proj = std::identity, class Comp = std::ranges::less>
concept HasEqualRangeIter =
    requires(Iter&& iter, Sent&& sent, const T& value, Comp&& comp, Proj&& proj) {
      std::ranges::equal_range(
          std::forward<Iter>(iter),
          std::forward<Sent>(sent),
          value,
          std::forward<Comp>(comp),
          std::forward<Proj>(proj));
    };

static_assert(HasEqualRangeIter<int*, int*, int>);

// !forward_iterator<I>
static_assert(!HasEqualRangeIter<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>, int>);

// !sentinel_for<S, I>
static_assert(!HasEqualRangeIter<int*, SentinelForNotSemiregular, int>);

// !indirect_strict_weak_order<Comp, const T*, projected<I, Proj>>
static_assert(!HasEqualRangeIter<int*, int*, Foo>);

template <class R, class T, class Proj = std::identity, class Comp = std::ranges::less>
concept HasEqualRangeForRange =
    requires(R&& r, const T& value, Comp&& comp, Proj&& proj) {
      std::ranges::equal_range(std::forward<R>(r), value, comp, proj);
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasEqualRangeForRange<R<int*>, int>);

// !forward_range<R>
static_assert(!HasEqualRangeForRange<R<cpp20_input_iterator<int*>>, int>);

// !indirect_strict_weak_order<Comp, const T*, projected<ranges::iterator_t<R>, Proj>>
static_assert(!HasEqualRangeForRange<R<int*>, Foo>);

template <class InIter, std::size_t N>
constexpr void testEqualRangeImpl(std::array<int, N>& in, int value, std::ranges::subrange<int*> expected) {
  using Sent = sentinel_wrapper<InIter>;

  // iterator overload
  {
    std::same_as<std::ranges::subrange<InIter, InIter>> decltype(auto) result =
        std::ranges::equal_range(InIter{in.data()}, Sent{InIter{in.data() + in.size()}}, value);

    assert(base(result.begin()) == expected.begin());
    assert(base(result.end()) == expected.end());
  }

  // range overload
  {
    std::ranges::subrange r{InIter{in.data()}, Sent{InIter{in.data() + in.size()}}};
    std::same_as<std::ranges::subrange<InIter, InIter>> decltype(auto) result = std::ranges::equal_range(r, value);

    assert(base(result.begin()) == expected.begin());
    assert(base(result.end()) == expected.end());
  }
}

template <class InIter>
constexpr void testImpl() {
  // no match and the searched value would be in the middle
  {
    std::array in{0, 1, 5, 6, 9, 10};
    int value = 7;
    std::ranges::subrange<int*> expected{in.data() + 4, in.data() + 4};
    testEqualRangeImpl<InIter>(in, value, expected);
  }

  // value smaller than all the elements
  {
    std::array in{0, 1, 5, 6, 9, 10};
    int value = -1;
    std::ranges::subrange<int*> expected{in.data(), in.data()};
    testEqualRangeImpl<InIter>(in, value, expected);
  }

  // value bigger than all the elements
  {
    std::array in{0, 1, 5, 6, 9, 10};
    int value = 20;
    std::ranges::subrange<int*> expected{in.data() + in.size(), in.data() + in.size()};
    testEqualRangeImpl<InIter>(in, value, expected);
  }

  // exactly one match
  {
    std::array in{0, 1, 5, 6, 9, 10};
    int value = 5;
    std::ranges::subrange<int*> expected{in.data() + 2, in.data() + 3};
    testEqualRangeImpl<InIter>(in, value, expected);
  }

  // multiple matches
  {
    std::array in{0, 1, 5, 6, 6, 6, 9, 10};
    int value = 6;
    std::ranges::subrange<int*> expected{in.data() + 3, in.data() + 6};
    testEqualRangeImpl<InIter>(in, value, expected);
  }

  // all matches
  {
    std::array in{6, 6, 6, 6, 6};
    int value = 6;
    std::ranges::subrange<int*> expected{in.data(), in.data() + in.size()};
    testEqualRangeImpl<InIter>(in, value, expected);
  }

  // empty range
  {
    std::array<int, 0> in{};
    int value = 6;
    std::ranges::subrange<int*> expected{in.data(), in.data()};
    testEqualRangeImpl<InIter>(in, value, expected);
  }

  // partially sorted
  {
    std::array in{3, 1, 5, 2, 6, 6, 10, 7, 8};
    int value = 6;
    std::ranges::subrange<int*> expected{in.data() + 4, in.data() + 6};
    testEqualRangeImpl<InIter>(in, value, expected);
  }
}

constexpr bool test() {
  testImpl<forward_iterator<int*>>();
  testImpl<bidirectional_iterator<int*>>();
  testImpl<random_access_iterator<int*>>();
  testImpl<contiguous_iterator<int*>>();

  struct Data {
    int data;
  };

  // Test custom comparator
  {
    std::array<Data, 10> in{{{2}, {1}, {3}, {3}, {3}, {10}, {9}, {5}, {5}, {7}}};
    Data value{3};

    const auto comp = [](const Data& x, const Data& y) { return x.data < y.data; };
    // iterator overload
    {
      auto result = std::ranges::equal_range(in.begin(), in.end(), value, comp);

      assert(result.begin() == in.begin() + 2);
      assert(result.end() == in.begin() + 5);
    }

    // range overload
    {
      auto result = std::ranges::equal_range(in, value, comp);

      assert(result.begin() == in.begin() + 2);
      assert(result.end() == in.begin() + 5);
    }
  }

  // Test custom projection
  {
    std::array<Data, 10> in{{{2}, {1}, {3}, {3}, {3}, {10}, {9}, {5}, {5}, {7}}};
    int value = 3;

    const auto proj = [](const Data& d) { return d.data; };
    // iterator overload
    {
      auto result = std::ranges::equal_range(in.begin(), in.end(), value, {}, proj);

      assert(result.begin() == in.begin() + 2);
      assert(result.end() == in.begin() + 5);
    }

    // range overload
    {
      auto result = std::ranges::equal_range(in, value, {}, proj);

      assert(result.begin() == in.begin() + 2);
      assert(result.end() == in.begin() + 5);
    }
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
