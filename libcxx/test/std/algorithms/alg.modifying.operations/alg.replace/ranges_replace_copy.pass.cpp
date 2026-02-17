//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<input_iterator I, sentinel_for<I> S, class T1, class T2,
//          output_iterator<const T2&> O, class Proj = identity>
//   requires indirectly_copyable<I, O> &&
//            indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T1*>
//   constexpr replace_copy_result<I, O>
//     replace_copy(I first, S last, O result, const T1& old_value, const T2& new_value,
//                  Proj proj = {});                                                                // Since C++20
//
// template<input_range R, class T1, class T2, output_iterator<const T2&> O,
//          class Proj = identity>
//   requires indirectly_copyable<iterator_t<R>, O> &&
//            indirect_binary_predicate<ranges::equal_to,
//                                      projected<iterator_t<R>, Proj>, const T1*>
//   constexpr replace_copy_result<borrowed_iterator_t<R>, O>
//     replace_copy(R&& r, O result, const T1& old_value, const T2& new_value,
//                  Proj proj = {});                                                                // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>
#include <utility>

#include "almost_satisfies_types.h"
#include "counting_projection.h"
#include "test_iterators.h"

template <class Iter, class Sent = sentinel_wrapper<Iter>, class OutIter = int*>
concept HasReplaceCopyIter =
  requires(Iter&& first, Sent&& last, OutIter&& result) {
    std::ranges::replace_copy(
        std::forward<Iter>(first), std::forward<Sent>(last), std::forward<OutIter>(result), 0, 0);
};

static_assert(HasReplaceCopyIter<int*>);

// !input_iterator<I>
static_assert(!HasReplaceCopyIter<InputIteratorNotDerivedFrom>);
static_assert(!HasReplaceCopyIter<InputIteratorNotIndirectlyReadable>);
static_assert(!HasReplaceCopyIter<InputIteratorNotInputOrOutputIterator>);

// !sentinel_for<S, I>
static_assert(!HasReplaceCopyIter<int*, SentinelForNotSemiregular>);
static_assert(!HasReplaceCopyIter<int*, SentinelForNotWeaklyEqualityComparableWith>);

// !output_iterator<O, const T2&>
static_assert(!HasReplaceCopyIter<int*, int*, OutputIteratorNotIndirectlyWritable>);
static_assert(!HasReplaceCopyIter<int*, int*, OutputIteratorNotInputOrOutputIterator>);

// !indirectly_copyable<I, O>
static_assert(!HasReplaceCopyIter<int*, int*, int**>);

// !indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T1*>
static_assert(!HasReplaceCopyIter<IndirectBinaryPredicateNotIndirectlyReadable>);

template <class Range, class OutIter = int*>
concept HasReplaceCopyRange = requires(Range&& range, OutIter&& result) {
  std::ranges::replace_copy(std::forward<Range>(range), std::forward<OutIter>(result), 0, 0);
};

template <class T>
using R = UncheckedRange<T>;

static_assert(HasReplaceCopyRange<R<int*>>);

// !input_range<R>
static_assert(!HasReplaceCopyRange<InputRangeNotDerivedFrom>);
static_assert(!HasReplaceCopyRange<InputRangeNotIndirectlyReadable>);
static_assert(!HasReplaceCopyRange<InputRangeNotInputOrOutputIterator>);
static_assert(!HasReplaceCopyRange<InputRangeNotSentinelSemiregular>);
static_assert(!HasReplaceCopyRange<InputRangeNotSentinelEqualityComparableWith>);

// !output_iterator<O, const T2&>
static_assert(!HasReplaceCopyRange<R<int*>, OutputIteratorNotIndirectlyWritable>);
static_assert(!HasReplaceCopyRange<R<int*>, OutputIteratorNotInputOrOutputIterator>);

// !indirectly_copyable<iterator_t<R>, O>
static_assert(!HasReplaceCopyRange<R<int*>, R<int**>>);

// !indirect_binary_predicate<ranges::equal_to, projected<iterator_t<T>, Proj>, const T1*>
static_assert(!HasReplaceCopyRange<R<IndirectBinaryPredicateNotIndirectlyReadable>>);

template <int N>
struct Data {
  std::array<int, N> input;
  int old_value;
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

    std::same_as<std::ranges::replace_copy_result<InIter, OutIter>> decltype(auto) ret =
        std::ranges::replace_copy(std::move(first), std::move(last), std::move(result), d.old_value, d.new_value);
    assert(base(ret.in) == d.input.data() + d.input.size());
    assert(base(ret.out) == output.data() + output.size());
    assert(d.expected == output);
  }

  { // range overload
    std::array<int, N> output;

    auto range  = std::ranges::subrange(InIter(d.input.data()), Sent(InIter(d.input.data() + d.input.size())));
    auto result = OutIter(output.data());

    std::same_as<std::ranges::replace_copy_result<InIter, OutIter>> decltype(auto) ret =
        std::ranges::replace_copy(range, result, d.old_value, d.new_value);
    assert(base(ret.in) == d.input.data() + d.input.size());
    assert(base(ret.out) == output.data() + output.size());
    assert(d.expected == output);
  }
}

template <class InIter, class Sent, class OutIter>
constexpr void tests() {
  // simple test
  test<InIter, Sent, OutIter, 4>({.input = {1, 2, 3, 4}, .old_value = 2, .new_value = 5, .expected = {1, 5, 3, 4}});
  // empty range
  test<InIter, Sent, OutIter, 0>({.input = {}, .old_value = 2, .new_value = 5, .expected = {}});
  // all elements match
  test<InIter, Sent, OutIter, 4>({.input = {1, 1, 1, 1}, .old_value = 1, .new_value = 2, .expected = {2, 2, 2, 2}});
  // no element matches
  test<InIter, Sent, OutIter, 4>({.input = {1, 1, 1, 1}, .old_value = 2, .new_value = 3, .expected = {1, 1, 1, 1}});
  // old_value and new_value are identical - match
  test<InIter, Sent, OutIter, 4>({.input = {1, 1, 1, 1}, .old_value = 1, .new_value = 1, .expected = {1, 1, 1, 1}});
  // old_value and new_value are identical - no match
  test<InIter, Sent, OutIter, 4>({.input = {1, 1, 1, 1}, .old_value = 2, .new_value = 2, .expected = {1, 1, 1, 1}});
  // more elements
  test<InIter, Sent, OutIter, 7>(
      {.input = {1, 2, 3, 4, 5, 6, 7}, .old_value = 2, .new_value = 3, .expected = {1, 3, 3, 4, 5, 6, 7}});
  // single element - match
  test<InIter, Sent, OutIter, 1>({.input = {1}, .old_value = 1, .new_value = 5, .expected = {5}});
  // single element - no match
  test<InIter, Sent, OutIter, 1>({.input = {2}, .old_value = 1, .new_value = 5, .expected = {2}});
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
      auto ret = std::ranges::replace_copy(std::begin(a), std::end(a), std::begin(b), 1, S{2}, &S::i);
      assert(ret.in == std::end(a));
      assert(ret.out == std::end(b));
    }

    { // range overload
      S a[] = {{1}, {2}, {3}, {4}};
      S b[4];
      auto ret = std::ranges::replace_copy(a, std::begin(b), 1, S{2}, &S::i);
      assert(ret.in == std::end(a));
      assert(ret.out == std::end(b));
    }
  }

  {   // Complexity: exactly `last - first` applications of the corresponding predicate and any projection.
    { // iterator overload
      int proj_count = 0;
      int a[] = {1, 2, 3, 4};
      int b[4];
      std::ranges::replace_copy(
          std::begin(a), std::end(a), std::begin(b), 0, 0, counting_projection(proj_count));
      assert(proj_count == 4);
    }

    { // range overload
      int proj_count = 0;
      int a[] = {1, 2, 3, 4};
      int b[4];
      std::ranges::replace_copy(a, std::begin(b), 0, 0, counting_projection(proj_count));
      assert(proj_count == 4);
    }
  }

  { // using different types for the old and new values works
    struct S {
      constexpr operator int() const { return 0; }
      constexpr bool operator==(const S&) const = default;
      constexpr bool operator==(int i) const { return i == 0; }
    };
    struct T {
      constexpr operator int() const { return 1; }
    };
    {
      int a[] = {0, 1, 2, 3};
      int b[4];
      std::ranges::replace_copy(std::begin(a), std::end(a), std::begin(b), S{}, T{});
      assert(std::ranges::equal(b, std::array{1, 1, 2, 3}));
    }
    {
      int a[] = {0, 1, 2, 3};
      int b[4];
      std::ranges::replace_copy(a, std::begin(b), S{}, T{});
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
