//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<input_iterator I, sentinel_for<I> S, weakly_incrementable O, class T,
//          class Proj = identity>
//   requires indirectly_copyable<I, O> &&
//            indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T*>
//   constexpr remove_copy_result<I, O>
//     remove_copy(I first, S last, O result, const T& value, Proj proj = {});                      // Since C++20
//
// template<input_range R, weakly_incrementable O, class T, class Proj = identity>
//   requires indirectly_copyable<iterator_t<R>, O> &&
//            indirect_binary_predicate<ranges::equal_to,
//                                      projected<iterator_t<R>, Proj>, const T*>
//   constexpr remove_copy_result<borrowed_iterator_t<R>, O>
//     remove_copy(R&& r, O result, const T& value, Proj proj = {});                                // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>
#include <utility>

#include "almost_satisfies_types.h"
#include "counting_projection.h"
#include "test_iterators.h"

struct ToPtr {
  int* operator()(int) const;
};

template <class Iter = int*, class Sent = int*, class OutIter = int*, class Proj = std::identity>
concept HasRemoveCopyIter =
  requires(Iter&& iter, Sent&& sent, OutIter&& out, Proj&& proj) {
    std::ranges::remove_copy(
        std::forward<Iter>(iter), std::forward<Sent>(sent), std::forward<OutIter>(out), 0, std::forward<Proj>(proj));
};

static_assert(HasRemoveCopyIter<int*>);

// !input_iterator<I>
static_assert(!HasRemoveCopyIter<InputIteratorNotDerivedFrom>);
static_assert(!HasRemoveCopyIter<cpp20_output_iterator<int*>>);

// !sentinel_for<S, I>
static_assert(!HasRemoveCopyIter<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasRemoveCopyIter<int*, SentinelForNotSemiregular>);

// !weakly_incrementable<O>
static_assert(!HasRemoveCopyIter<int*, int*, WeaklyIncrementableNotMovable>);

// !indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T*>
static_assert(!HasRemoveCopyIter<int*, int*, int*, ToPtr>);

// !indirectly_copyable<I, O>
static_assert(!HasRemoveCopyIter<int*, int*, OutputIteratorNotIndirectlyWritable>);
static_assert(!HasRemoveCopyIter<const int*, const int*, const int*>);

template <class Range, class OutIter = int*, class Proj = std::identity>
concept HasRemoveCopyRange =
  requires(Range&& range, OutIter&& out, Proj&& proj) {
    std::ranges::remove_copy(
        std::forward<Range>(range), std::forward<OutIter>(out), 0, std::forward<Proj>(proj));
};

template <class T>
using R = UncheckedRange<T>;

static_assert(HasRemoveCopyRange<R<int*>>);

// !input_range<R>
static_assert(!HasRemoveCopyRange<InputRangeNotDerivedFrom>);
static_assert(!HasRemoveCopyRange<InputRangeNotIndirectlyReadable>);
static_assert(!HasRemoveCopyRange<InputRangeNotInputOrOutputIterator>);
static_assert(!HasRemoveCopyRange<InputRangeNotSentinelSemiregular>);
static_assert(!HasRemoveCopyRange<InputRangeNotSentinelEqualityComparableWith>);

// !weakly_incrementable<O>
static_assert(!HasRemoveCopyRange<R<int*>, WeaklyIncrementableNotMovable>);

// !indirect_binary_predicate<ranges::equal_to, projected<iterator_t<I>, Proj>, const T*>
static_assert(!HasRemoveCopyRange<R<int*>, int*, ToPtr>);

// !indirectly_copyable<I, O>
static_assert(!HasRemoveCopyRange<R<int*>, int*, OutputIteratorNotIndirectlyWritable>);
static_assert(!HasRemoveCopyRange<const int*, const int*, const int*>);

template <int N, int M>
struct Data {
  std::array<int, N> input;
  std::array<int, M> expected;
  int val;
};

template <class InIter, class Sent, class OutIter, int N, int M>
constexpr void test(Data<N, M> d) {
  using Result = std::ranges::remove_copy_result<InIter, OutIter>;

  { // iterator overload
    std::array<int, M> output;

    std::same_as<Result> decltype(auto) ret = std::ranges::remove_copy(
        InIter(d.input.data()), Sent(InIter(d.input.data() + d.input.size())), OutIter(output.data()), d.val);

    assert(base(ret.in) == d.input.data() + N);
    assert(base(ret.out) == output.data() + M);
    assert(d.expected == output);
  }

  { // range overload
    std::array<int, M> output;
    auto range = std::ranges::subrange(InIter(d.input.data()), Sent(InIter(d.input.data() + d.input.size())));

    std::same_as<Result> decltype(auto) ret =
        std::ranges::remove_copy(range, OutIter(output.data()), d.val);

    assert(base(ret.in) == d.input.data() + N);
    assert(base(ret.out) == output.data() + M);
    assert(d.expected == output);
  }
}

template <class Iter, class Sent, class OutIter>
constexpr void tests() {
  // simple test
  test<Iter, Sent, OutIter, 6, 5>({.input = {1, 2, 3, 4, 5, 6}, .expected = {1, 2, 3, 4, 6}, .val = 5});
  // empty range
  test<Iter, Sent, OutIter, 0, 0>({});
  // single element range - match
  test<Iter, Sent, OutIter, 1, 0>({.input = {1}, .expected = {}, .val = 1});
  // single element range - no match
  test<Iter, Sent, OutIter, 1, 1>({.input = {1}, .expected = {1}, .val = 2});
  // two element range - same order
  test<Iter, Sent, OutIter, 2, 1>({.input = {1, 2}, .expected = {1}, .val = 2});
  // two element range - reversed order
  test<Iter, Sent, OutIter, 2, 1>({.input = {1, 2}, .expected = {2}, .val = 1});
  // all elements match
  test<Iter, Sent, OutIter, 5, 0>({.input = {1, 1, 1, 1, 1}, .expected = {}, .val = 1});
  // the relative order of elements isn't changed
  test<Iter, Sent, OutIter, 8, 5>({.input = {1, 2, 3, 2, 3, 4, 2, 5}, .expected = {1, 3, 3, 4, 5}, .val = 2});
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

template <class Iter>
constexpr void test_sentinels() {
  test_output_iterators<Iter, Iter>();
  test_output_iterators<Iter, sentinel_wrapper<Iter>>();
  test_output_iterators<Iter, sized_sentinel<Iter>>();
}

constexpr bool test() {
  test_output_iterators<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_output_iterators<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>();
  test_output_iterators<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_output_iterators<cpp20_input_iterator<int*>, sized_sentinel<cpp20_input_iterator<int*>>>();
  test_sentinels<forward_iterator<int*>>();
  test_sentinels<bidirectional_iterator<int*>>();
  test_sentinels<random_access_iterator<int*>>();
  test_sentinels<contiguous_iterator<int*>>();
  test_sentinels<int*>();

  { // check that passing a different type works
    struct S {
      constexpr operator int() const { return 3; }
    };

    { // iterator overload
      int a[] = {1, 2, 3, 4};
      int b[3];
      std::ranges::remove_copy(std::begin(a), std::end(a), std::begin(b), S{});
    }

    { // range overload
      int a[] = {1, 2, 3, 4};
      int b[3];
      std::ranges::remove_copy(a, std::begin(b), S{});
    }
  }

  { // check that a custom projection works
    struct S {
      constexpr operator int() const { return 3; }
    };

    { // iterator overload
      int a[] = {1, 2, 3, 4};
      int b[3];
      std::ranges::remove_copy(std::begin(a), std::end(a), std::begin(b), S{});

    }
    { // range overload
      int a[] = {1, 2, 3, 4};
      int b[3];
      std::ranges::remove_copy(a, std::begin(b), S{});
    }
  }

  // Complexity: Exactly last - first applications of the corresponding predicate and any projection.

  {
    std::array in{4, 4, 5, 6};
    std::array expected{5, 6};

    // iterator overload
    {
      int numberOfProj = 0;
      std::array<int, 2> out;
      std::ranges::remove_copy(
          in.begin(),
          in.end(),
          out.begin(),
          4,
          counting_projection(numberOfProj));

      assert(numberOfProj == static_cast<int>(in.size()));

      assert(std::ranges::equal(out, expected));
    }

    // range overload
    {
      int numberOfProj = 0;
      std::array<int, 2> out;
      std::ranges::remove_copy(
          in, out.begin(), 4, counting_projection(numberOfProj));
      assert(numberOfProj == static_cast<int>(in.size()));
      assert(std::ranges::equal(out, expected));
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
