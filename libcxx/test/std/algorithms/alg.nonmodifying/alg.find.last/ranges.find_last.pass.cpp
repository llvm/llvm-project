//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ADDITIONAL_COMPILE_FLAGS(gcc-style-warnings): -Wno-sign-compare
// MSVC warning C4242: 'argument': conversion from 'const _Ty' to 'ElementT', possible loss of data
// MSVC warning C4244: 'argument': conversion from 'const _Ty' to 'ElementT', possible loss of data
// ADDITIONAL_COMPILE_FLAGS(cl-style-warnings): /wd4242 /wd4244

// template<forward_iterator I, sentinel_for<I> S, class T, class Proj = identity>
//   requires indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T*>
//   constexpr subrange<I> ranges::find_last(I first, S last, const T& value, Proj proj = {});
// template<forward_range R, class T, class Proj = identity>
//   requires indirect_binary_predicate<ranges::equal_to, projected<iterator_t<R>, Proj>, const T*>
//   constexpr borrowed_subrange_t<R> ranges::find_last(R&& r, const T& value, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <ranges>
#include <vector>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct NotEqualityComparable {};

template <class It, class Sent = It>
concept HasFindLastIt = requires(It it, Sent sent) { std::ranges::find_last(it, sent, *it); };
static_assert(HasFindLastIt<int*>);
static_assert(HasFindLastIt<forward_iterator<int*>>);
static_assert(!HasFindLastIt<cpp20_input_iterator<int*>>);
static_assert(!HasFindLastIt<NotEqualityComparable*>);
static_assert(!HasFindLastIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasFindLastIt<ForwardIteratorNotIncrementable>);
static_assert(!HasFindLastIt<forward_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasFindLastIt<forward_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);

static_assert(!HasFindLastIt<int*, int>);
static_assert(!HasFindLastIt<int, int*>);

template <class Range, class ValT>
concept HasFindLastR = requires(Range r) { std::ranges::find_last(r, ValT{}); };
static_assert(HasFindLastR<std::array<int, 1>, int>);
static_assert(!HasFindLastR<int, int>);
static_assert(!HasFindLastR<std::array<NotEqualityComparable, 1>, NotEqualityComparable>);
static_assert(!HasFindLastR<ForwardRangeNotDerivedFrom, int>);
static_assert(!HasFindLastR<ForwardRangeNotIncrementable, int>);
static_assert(!HasFindLastR<ForwardRangeNotSentinelSemiregular, int>);
static_assert(!HasFindLastR<ForwardRangeNotSentinelEqualityComparableWith, int>);

template <class It, class Sent = It>
constexpr void test_iterators() {
  using ValueT    = std::iter_value_t<It>;
  auto make_range = [](auto& a) {
    return std::ranges::subrange(
        It(std::to_address(std::ranges::begin(a))), Sent(It(std::to_address(std::ranges::end(a)))));
  };
  { // simple test
    {
      ValueT a[] = {1, 2, 3, 4};

      std::same_as<std::ranges::subrange<It>> auto ret = std::ranges::find_last(It(a), Sent(It(a + 4)), 2);
      assert(base(ret.begin()) == a + 1);
      assert(*ret.begin() == 2);
    }
    {
      ValueT a[] = {1, 2, 3, 4};

      std::same_as<std::ranges::subrange<It>> auto ret = std::ranges::find_last(make_range(a), 2);
      assert(base(ret.begin()) == a + 1);
      assert(*ret.begin() == 2);
    }
  }

  { // check that an empty range works
    {
      std::array<ValueT, 0> a = {};

      auto ret = std::ranges::find_last(It(a.data()), Sent(It(a.data())), 1).begin();
      assert(ret == It(a.data()));
    }
    {
      std::array<ValueT, 0> a = {};

      auto ret = std::ranges::find_last(make_range(a), 1).begin();
      assert(ret == It(a.data()));
    }
  }

  { // check that last is returned with no match
    {
      ValueT a[] = {1, 1, 1};

      auto ret = std::ranges::find_last(It(a), Sent(It(a + 3)), 0).begin();
      assert(ret == It(a + 3));
    }
    {
      ValueT a[] = {1, 1, 1};

      auto ret = std::ranges::find_last(make_range(a), 0).begin();
      assert(ret == It(a + 3));
    }
  }
}

template <template <class> class IteratorT>
constexpr void test_iterator_classes() {
  { // check that the last element is returned
    struct S {
      int comp;
      int other;
    };
    using it = IteratorT<S*>;
    S a[]    = {{0, 0}, {0, 2}, {0, 1}};

    auto ret = std::ranges::find_last(it(std::begin(a)), it(std::end(a)), 0, &S::comp).begin();
    assert(ret == it(a + 2));
    assert((*ret).comp == 0);
    assert((*ret).other == 1);
  }

  {
    // count invocations of the projection
    using it = IteratorT<int*>;

    int a[]              = {1, 2, 3, 4};
    int projection_count = 0;

    auto ret = std::ranges::find_last(it(std::begin(a)), it(std::end(a)), 2, [&](int i) {
                 ++projection_count;
                 return i;
               }).begin();
    assert(ret == it(a + 1));
    assert(*ret == 2);
    if (std::bidirectional_iterator<it>) {
      assert(projection_count == 3);
    } else {
      assert(projection_count == 4); // must go through entire list
    }
  }
}

template <class ElementT>
class TriviallyComparable {
  ElementT el_;

public:
  constexpr TriviallyComparable(ElementT el) : el_(el) {}
  bool operator==(const TriviallyComparable&) const = default;
};

constexpr bool test() {
  types::for_each(types::type_list<char, int, TriviallyComparable<char>>{}, []<class T> {
    types::for_each(types::forward_iterator_list<T*>{}, []<class Iter> {
      test_iterators<Iter>();
      test_iterators<Iter, sentinel_wrapper<Iter>>();
      test_iterators<Iter, sized_sentinel<Iter>>();
    });
  });

  test_iterator_classes<forward_iterator>();
  test_iterator_classes<bidirectional_iterator>();
  test_iterator_classes<random_access_iterator>();
  test_iterator_classes<std::type_identity_t>();

  {
    std::vector<std::vector<int>> vec = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto view = vec | std::views::join;
    assert(std::ranges::find_last(view.begin(), view.end(), 4).begin() == std::next(view.begin(), 3));
    assert(std::ranges::find_last(view, 4).begin() == std::next(view.begin(), 3));
  }

  {
    // check that an iterator is returned with a borrowing range
    int a[] = {1, 2, 3, 4};

    std::same_as<std::ranges::subrange<int*>> auto ret = std::ranges::find_last(std::views::all(a), 1);
    assert(ret.begin() == a);
    assert(*ret.begin() == 1);
  }

  {
    // check that dangling ranges are dangling
    std::same_as<std::ranges::dangling> auto ret = std::ranges::find_last(std::vector<int>(), 0);
    (void)ret;
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
