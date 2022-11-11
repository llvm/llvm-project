//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<input_iterator I, sentinel_for<I> S, class T, output_iterator<const T&> O,
//          class Proj = identity, indirect_unary_predicate<projected<I, Proj>> Pred>
//   requires indirectly_copyable<I, O>
//   constexpr replace_copy_if_result<I, O>
//     replace_copy_if(I first, S last, O result, Pred pred, const T& new_value,
//                     Proj proj = {});                                                             // Since C++20
//
// template<input_range R, class T, output_iterator<const T&> O, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   requires indirectly_copyable<iterator_t<R>, O>
//   constexpr replace_copy_if_result<borrowed_iterator_t<R>, O>
//     replace_copy_if(R&& r, O result, Pred pred, const T& new_value,
//                     Proj proj = {});                                                             // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>
#include <utility>

#include "almost_satisfies_types.h"
#include "counting_predicates.h"
#include "counting_projection.h"
#include "test_iterators.h"

struct FalsePredicate {
  constexpr bool operator()(int) { return false; }
};

template <class Iter, class Sent = sentinel_wrapper<Iter>, class OutIter = int*>
concept HasReplaceCopyIfIter = requires(Iter&& first, Sent&& last, OutIter&& result) {
  std::ranges::replace_copy_if(
      std::forward<Iter>(first), std::forward<Sent>(last), std::forward<OutIter>(result), FalsePredicate{}, 0);
};

static_assert(HasReplaceCopyIfIter<int*>);

// !input_iterator<I>
static_assert(!HasReplaceCopyIfIter<InputIteratorNotDerivedFrom>);
static_assert(!HasReplaceCopyIfIter<InputIteratorNotIndirectlyReadable>);
static_assert(!HasReplaceCopyIfIter<InputIteratorNotInputOrOutputIterator>);

// !sentinel_for<S, I>
static_assert(!HasReplaceCopyIfIter<int*, SentinelForNotSemiregular>);
static_assert(!HasReplaceCopyIfIter<int*, SentinelForNotWeaklyEqualityComparableWith>);

// !output_iterator<O, const T2&>
static_assert(!HasReplaceCopyIfIter<int*, int*, OutputIteratorNotIndirectlyWritable>);
static_assert(!HasReplaceCopyIfIter<int*, int*, OutputIteratorNotInputOrOutputIterator>);

// !indirect_unary_predicate<Pred, projected<I, Proj>> Pred>
static_assert(!HasReplaceCopyIfIter<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasReplaceCopyIfIter<IndirectUnaryPredicateNotPredicate>);

// !indirectly_copyable<I, O>
static_assert(!HasReplaceCopyIfIter<int*, int*, int**>);

template <class Range, class OutIter = int*>
concept HasReplaceCopyIfRange = requires(Range&& range, OutIter&& result) {
  std::ranges::replace_copy_if(std::forward<Range>(range), std::forward<OutIter>(result), FalsePredicate{}, 0);
};

template <class T>
using R = UncheckedRange<T>;

static_assert(HasReplaceCopyIfRange<R<int*>>);

// !input_range<R>
static_assert(!HasReplaceCopyIfRange<InputRangeNotDerivedFrom>);
static_assert(!HasReplaceCopyIfRange<InputRangeNotIndirectlyReadable>);
static_assert(!HasReplaceCopyIfRange<InputRangeNotInputOrOutputIterator>);
static_assert(!HasReplaceCopyIfRange<InputRangeNotSentinelSemiregular>);
static_assert(!HasReplaceCopyIfRange<InputRangeNotSentinelEqualityComparableWith>);

// !output_iterator<O, const T2&>
static_assert(!HasReplaceCopyIfRange<R<int*>, OutputIteratorNotIndirectlyWritable>);
static_assert(!HasReplaceCopyIfRange<R<int*>, OutputIteratorNotInputOrOutputIterator>);

// !indirect_unary_predicate<Pred, projected<iterator_t<R>, Proj>> Pred>
static_assert(!HasReplaceCopyIfRange<R<IndirectUnaryPredicateNotPredicate>>);

// !indirectly_copyable<iterator_t<R>, O>
static_assert(!HasReplaceCopyIfRange<R<int*>, int**>);

template <int N>
struct Data {
  std::array<int, N> input;
  int cutoff;
  int new_value;
  std::array<int, N> expected;
};

template <class InIter, class Sent, class OutIter, int N>
constexpr void test(Data<N> d) {
  { // iterator overload
    std::array<int, N> output;

    auto first  = InIter(d.input.data());
    auto last   = Sent(InIter(d.input.data() + d.input.size()));
    auto result = OutIter(output.data());

    auto pred = [&](int i) { return i < d.cutoff; };

    std::same_as<std::ranges::replace_copy_if_result<InIter, OutIter>> decltype(auto) ret =
        std::ranges::replace_copy_if(std::move(first), std::move(last), std::move(result), pred, d.new_value);
    assert(base(ret.in) == d.input.data() + d.input.size());
    assert(base(ret.out) == output.data() + output.size());
    assert(d.expected == output);
  }

  { // range overload
    std::array<int, N> output;

    auto range  = std::ranges::subrange(InIter(d.input.data()), Sent(InIter(d.input.data() + d.input.size())));
    auto result = OutIter(output.data());

    auto pred = [&](int i) { return i < d.cutoff; };

    std::same_as<std::ranges::replace_copy_if_result<InIter, OutIter>> decltype(auto) ret =
        std::ranges::replace_copy_if(range, result, pred, d.new_value);
    assert(base(ret.in) == d.input.data() + d.input.size());
    assert(base(ret.out) == output.data() + output.size());
    assert(d.expected == output);
  }
}

template <class InIter, class Sent, class OutIter>
constexpr void tests() {
  // simple test
  test<InIter, Sent, OutIter, 4>({.input = {1, 2, 3, 4}, .cutoff = 2, .new_value = 5, .expected = {5, 2, 3, 4}});
  // empty range
  test<InIter, Sent, OutIter, 0>({.input = {}, .cutoff = 2, .new_value = 5, .expected = {}});
  // all elements match
  test<InIter, Sent, OutIter, 4>({.input = {1, 1, 1, 1}, .cutoff = 2, .new_value = 2, .expected = {2, 2, 2, 2}});
  // no element matches
  test<InIter, Sent, OutIter, 4>({.input = {1, 1, 1, 1}, .cutoff = 1, .new_value = 3, .expected = {1, 1, 1, 1}});
  // more elements
  test<InIter, Sent, OutIter, 7>(
      {.input = {1, 2, 3, 4, 5, 6, 7}, .cutoff = 3, .new_value = 3, .expected = {3, 3, 3, 4, 5, 6, 7}});
  // single element - match
  test<InIter, Sent, OutIter, 1>({.input = {1}, .cutoff = 2, .new_value = 5, .expected = {5}});
  // single element - no match
  test<InIter, Sent, OutIter, 1>({.input = {2}, .cutoff = 2, .new_value = 5, .expected = {2}});
}

template <class InIter, class Sent>
constexpr void test_output_iterators() {
  tests<InIter, Sent, cpp17_output_iterator<int*>>();
  tests<InIter, Sent, forward_iterator<int*>>();
  tests<InIter, Sent, bidirectional_iterator<int*>>();
  tests<InIter, Sent, random_access_iterator<int*>>();
  tests<InIter, Sent, contiguous_iterator<int*>>();
  tests<InIter, Sent, int*>();
}

template <class InIter>
constexpr void test_sentinels() {
  test_output_iterators<InIter, InIter>();
  test_output_iterators<InIter, sentinel_wrapper<InIter>>();
  test_output_iterators<InIter, sized_sentinel<InIter>>();
}

constexpr bool test() {
  test_output_iterators<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_output_iterators<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_sentinels<forward_iterator<int*>>();
  test_sentinels<bidirectional_iterator<int*>>();
  test_sentinels<random_access_iterator<int*>>();
  test_sentinels<contiguous_iterator<int*>>();
  test_sentinels<int*>();
  test_sentinels<const int*>();

  { // check that a custom projection works
    struct S {
      int i;
    };

    { // iterator overload
      S a[] = {{1}, {2}, {3}, {4}};
      S b[4];
      auto ret = std::ranges::replace_copy_if(std::begin(a), std::end(a), std::begin(b), FalsePredicate{}, S{2}, &S::i);
      assert(ret.in == std::end(a));
      assert(ret.out == std::end(b));
      assert(std::ranges::equal(a, b, {}, &S::i, &S::i));
    }

    { // range overload
      S a[] = {{1}, {2}, {3}, {4}};
      S b[4];
      auto ret = std::ranges::replace_copy_if(a, std::begin(b), FalsePredicate{}, S{2}, &S::i);
      assert(ret.in == std::end(a));
      assert(ret.out == std::end(b));
      assert(std::ranges::equal(a, b, {}, &S::i, &S::i));
    }
  }

  {   // Complexity: exactly `last - first` applications of the corresponding predicate and any projection.
    { // iterator overload
      int pred_count = 0;
      int proj_count = 0;
      int a[] = {1, 2, 3, 4};
      int b[4];

      std::ranges::replace_copy_if(
          std::begin(a), std::end(a), std::begin(b),
          counting_predicate(FalsePredicate{}, pred_count), 0, counting_projection(proj_count));
      assert(pred_count == 4);
      assert(proj_count == 4);
    }

    { // range overload
      int pred_count = 0;
      int proj_count = 0;
      int a[] = {1, 2, 3, 4};
      int b[4];

      std::ranges::replace_copy_if(a, std::begin(b),
          counting_predicate(FalsePredicate{}, pred_count), 0, counting_projection(proj_count));
      assert(pred_count == 4);
      assert(proj_count == 4);
    }
  }

  { // using different types for the old and new values works
    struct S {
      constexpr operator int() const { return 1; }
    };
    {
      int a[] = {0, 0, 2, 3};
      int b[4];
      std::ranges::replace_copy_if(std::begin(a), std::end(a), std::begin(b), [](int i) { return i < 2; }, S{});
      assert(std::ranges::equal(b, std::array{1, 1, 2, 3}));
    }

    {
      int a[] = {0, 0, 2, 3};
      int b[4];
      std::ranges::replace_copy_if(a, std::begin(b), [](int i) { return i < 2; }, S{});
      assert(std::ranges::equal(b, std::array{1, 1, 2, 3}));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
