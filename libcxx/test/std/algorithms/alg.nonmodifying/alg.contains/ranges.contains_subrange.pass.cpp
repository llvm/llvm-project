//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=2000000

// template<input_iterator I, sentinel_for<I> S, class T, class Proj = identity>
//     requires indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T*>
//     constexpr bool ranges::contains(I first, S last, const T& value, Proj proj = {});       // since C++23

// template<input_range R, class T, class Proj = identity>
//     requires indirect_binary_predicate<ranges::equal_to, projected<iterator_t<R>, Proj>, const T*>
//     constexpr bool ranges::contains(R&& r, const T& value, Proj proj = {});                 // since C++23

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>
#include <vector>

#include "almost_satisfies_types.h"
#include "boolean_testable.h"
#include "test_iterators.h"

struct NotEqualityComparable {};

template <class Iter1, class Sent1 = Iter1, class Iter2 = int*, class Sent2 = Iter2>
concept HasContainsSubrangeSubrangeIt = requires(Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2) {
  std::ranges::contains_subrange(first1, last1, first2, last2);
};

static_assert(HasContainsSubrangeSubrangeIt<int*>);
static_assert(!HasContainsSubrangeSubrangeIt<NotEqualityComparable*>);
static_assert(!HasContainsSubrangeSubrangeIt<InputIteratorNotDerivedFrom>);
static_assert(!HasContainsSubrangeSubrangeIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasContainsSubrangeSubrangeIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasContainsSubrangeSubrangeIt<cpp20_input_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasContainsSubrangeSubrangeIt<cpp20_input_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);
static_assert(!HasContainsSubrangeSubrangeIt<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>);

static_assert(!HasContainsSubrangeSubrangeIt<int*, int>);
static_assert(!HasContainsSubrangeSubrangeIt<int, int*>);
static_assert(HasContainsSubrangeSubrangeIt<int*, int*>);

template <class Range1, class Range2 = UncheckedRange<int*>>
concept HasContainsSubrangeR = requires(Range1&& range1, Range2&& range2) {
    std::ranges::contains_subrange(std::forward<Range1>(range1), std::forward<Range2>(range2)); };

static_assert(HasContainsSubrangeR<UncheckedRange<int*>>);
static_assert(HasContainsSubrangeR<ForwardRangeNotDerivedFrom>);
static_assert(!HasContainsSubrangeR<ForwardIteratorNotIncrementable>);
static_assert(!HasContainsSubrangeR<ForwardRangeNotSentinelSemiregular>);
static_assert(!HasContainsSubrangeR<ForwardRangeNotSentinelEqualityComparableWith>);
static_assert(!HasContainsSubrangeR<UncheckedRange<int*>, UncheckedRange<int**>>); // not indirectly comparable
static_assert(HasContainsSubrangeR<UncheckedRange<int*>, ForwardRangeNotDerivedFrom>);
static_assert(HasContainsSubrangeR<UncheckedRange<int*>, ForwardRangeNotIncrementable>);
static_assert(!HasContainsSubrangeR<UncheckedRange<int*>, ForwardRangeNotSentinelSemiregular>);
static_assert(!HasContainsSubrangeR<UncheckedRange<int*>, ForwardRangeNotSentinelEqualityComparableWith>);

static std::vector<int> comparable_data;

template <class Iter1, class Sent1 = Iter1, class Iter2, class Sent2 = Iter2>
constexpr void test_iterators() {
  {  // simple tests
    int a[] = {1, 2, 3, 4, 5, 6};
    int p[] = {3, 4, 5};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
    {
      [[maybe_unused]] std::same_as<bool> decltype(auto) ret =
        std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(ret);
    }
    {
      [[maybe_unused]] std::same_as<bool> decltype(auto) ret =
        std::ranges::contains_subrange(whole, subrange);
      assert(ret);
    }
  }

  { // no match
    int a[] = {1, 2, 3, 4, 5, 6};
    int p[] = {3, 4, 2};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(!ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(!ret);
    }
  }

  { // range consists of just one element
    int a[] = {3};
    int p[] = {3, 4, 2};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 1)));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(!ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(!ret);
    }
  }

  { // subrange consists of just one element
    int a[] = {23, 1, 20, 3, 54, 2};
    int p[] = {3};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 1)));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(ret);
    }
  }

  { // range has zero length
    int a[] = {};
    int p[] = {3, 4, 2};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a)));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(!ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(!ret);
    }
  }

  { // subrange has zero length
    int a[] = {3, 4, 2};
    int p[] = {};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 3)));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(p)));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(ret);
    }
  }

  { // range and subrange are identical
    int a[] = {3, 4, 11, 32, 54, 2};
    int p[] = {3, 4, 11, 32, 54, 2};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 6)));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(ret);
    }
  }

  { // subrange is longer than range
    int a[] = {3, 4, 2};
    int p[] = {23, 3, 4, 2, 11, 32, 54, 2};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 3)));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 8)));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(!ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(!ret);
    }
  }

  { // subrange is subsequence
    int a[] = {23, 1, 0, 54, 2};
    int p[] = {1, 0, 2};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 5)));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(!ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(!ret);
    }
  }

  { // repeated subrange
    int a[] = {23, 1, 0, 2, 54, 1, 0, 2, 23, 33};
    int p[] = {1, 0, 2};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 10)));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(ret);
    }
  }

  { // check that the predicate is used
    int a[] = {23, 81, 61, 0, 42, 25, 1, 2, 1, 29, 2};
    int p[] = {-1, -2, -1};
    auto pred = [](int l, int r) { return l * -1 == r; };
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 11)));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end(), pred);
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange, pred);
      assert(ret);
    }
  }

  { // check that the projections are used
    int a[] = {1, 3, 15, 1, 2, 1, 8};
    int p[] = {2, 1, 2};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 7)));
    auto subrange  = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end(), {},
         [](int i) { return i - 3; },
         [](int i) { return i * -1; });
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange, {},
         [](int i) { return i - 3; },
         [](int i) { return i * -1; });
      assert(ret);
    }
  }

  { // check the nodiscard extension
    // use #pragma around to suppress error: ignoring return value of function
    // declared with 'nodiscard' attribute [-Werror,-Wunused-result]
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-result"
    int a[] = {1, 9, 0, 13, 25};
    int p[] = {1, 9, 0};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 5)));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
    std::ranges::contains_subrange(whole, subrange);
    #pragma clang diagnostic pop
  }
}

constexpr bool test() {
    types::for_each(types::cpp20_input_iterator_list<int*>{}, []<class Iter2>() {
    types::for_each(types::cpp20_input_iterator_list<int*>{}, []<class Iter1>() {
      if constexpr (std::forward_iterator<Iter1> && std::forward_iterator<Iter2>)
        test_iterators<Iter1, Iter1, Iter2, Iter2>();
      if constexpr (std::forward_iterator<Iter2>)
        test_iterators<Iter1, sized_sentinel<Iter1>, Iter2, Iter2>();
      if constexpr (std::forward_iterator<Iter1>)
        test_iterators<Iter1, Iter1, Iter2, sized_sentinel<Iter2>>();
      test_iterators<Iter1, sized_sentinel<Iter1>, Iter2, sized_sentinel<Iter2>>();
    });
  });

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}