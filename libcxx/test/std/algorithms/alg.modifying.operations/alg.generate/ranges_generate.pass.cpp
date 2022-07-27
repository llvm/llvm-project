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

// template<input_or_output_iterator O, sentinel_for<O> S, copy_constructible F>
//   requires invocable<F&> && indirectly_writable<O, invoke_result_t<F&>>
//   constexpr O generate(O first, S last, F gen);                                                  // Since C++20
//
// template<class R, copy_constructible F>
//   requires invocable<F&> && output_range<R, invoke_result_t<F&>>
//   constexpr borrowed_iterator_t<R> generate(R&& r, F gen);                                       // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>
#include <utility>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct IntGen {
  int operator()() const;
};

struct UncopyableGen {
  UncopyableGen(const UncopyableGen&) = delete;
  int operator()() const;
};
static_assert(!std::copy_constructible<UncopyableGen>);
static_assert(std::invocable<UncopyableGen>);

struct UninvocableGen {
};
static_assert(std::copy_constructible<UninvocableGen>);
static_assert(!std::invocable<UninvocableGen>);

struct IntPtrGen {
  int* operator()() const;
};

// Test constraints of the (iterator, sentinel) overload.
// ======================================================

template <class Iter = int*, class Sent = int*, class Gen = IntGen>
concept HasGenerateIter =
    requires(Iter&& iter, Sent&& sent, Gen&& gen) {
      std::ranges::generate(std::forward<Iter>(iter), std::forward<Sent>(sent), std::forward<Gen>(gen));
    };

static_assert(HasGenerateIter<int*, int*, IntGen>);

// !input_or_output_iterator<O>
static_assert(!HasGenerateIter<InputIteratorNotInputOrOutputIterator>);

// !sentinel_for<S, O>
static_assert(!HasGenerateIter<int*, SentinelForNotSemiregular>);
static_assert(!HasGenerateIter<int*, SentinelForNotWeaklyEqualityComparableWith>);

// !copy_constructible<F>
static_assert(!HasGenerateIter<int*, int*, UncopyableGen>);

// !invocable<F&>
static_assert(!HasGenerateIter<int*, int*, UninvocableGen>);

// !indirectly_writable<O, invoke_result_t<F&>>
static_assert(!HasGenerateIter<int*, int*, IntPtrGen>);

// Test constraints of the (range) overload.
// =========================================

template <class Range, class Gen = IntGen>
concept HasGenerateRange =
    requires(Range&& range, Gen&& gen) {
      std::ranges::generate(std::forward<Range>(range), std::forward<Gen>(gen));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasGenerateRange<R<int*>, IntGen>);

// !copy_constructible<F>
static_assert(!HasGenerateRange<R<int*>, UncopyableGen>);

// !invocable<F&>
static_assert(!HasGenerateRange<R<int*>, UninvocableGen>);

// !output_range<R, invoke_result_t<F&>>
static_assert(!HasGenerateRange<InputRangeNotInputOrOutputIterator>);
static_assert(!HasGenerateRange<R<int*>, IntPtrGen>);

template <class Iter, class Sent, size_t N, class Gen>
constexpr void test_one(const std::array<int, N> input, Gen gen, std::array<int, N> expected) {
  { // (iterator, sentinel) overload.
    auto in = input;
    auto begin = Iter(in.data());
    auto end = Sent(Iter(in.data() + in.size()));

    std::same_as<Iter> decltype(auto) result = std::ranges::generate(std::move(begin), std::move(end), gen);
    assert(base(result) == in.data() + in.size());
    assert(in == expected);
  }

  { // (range) overload.
    auto in = input;
    auto begin = Iter(in.data());
    auto end = Sent(Iter(in.data() + in.size()));
    auto range = std::ranges::subrange(std::move(begin), std::move(end));

    // For some reason `ranges::generate` accepts both input and output iterators but only output (not input) ranges.
    if constexpr (std::ranges::output_range<decltype(range), std::invoke_result_t<Gen&>>) {
      std::same_as<Iter> decltype(auto) result = std::ranges::generate(std::move(range), gen);
      assert(base(result) == in.data() + in.size());
      assert(in == expected);
    }
  }
}

template <class Iter, class Sent>
constexpr void test_iter_sent() {
  auto gen = [ctr = 1] () mutable { return ctr++; };

  // Empty sequence.
  test_one<Iter, Sent, 0>({}, gen, {});
  // 1-element sequence.
  test_one<Iter, Sent>(std::array{-10}, gen, {1});
  // Longer sequence.
  test_one<Iter, Sent>(std::array<int, 5>{}, gen, {1, 2, 3, 4, 5});
}

template <class Iter>
constexpr void test_iter() {
  if constexpr (std::sentinel_for<Iter, Iter>) {
    test_iter_sent<Iter, Iter>();
  }
  test_iter_sent<Iter, sentinel_wrapper<Iter>>();
}

constexpr void test_iterators() {
  test_iter<cpp17_input_iterator<int*>>();
  test_iter<cpp20_input_iterator<int*>>();
  test_iter<cpp17_output_iterator<int*>>();
  test_iter<cpp20_output_iterator<int*>>();
  test_iter<forward_iterator<int*>>();
  test_iter<bidirectional_iterator<int*>>();
  test_iter<random_access_iterator<int*>>();
  test_iter<contiguous_iterator<int*>>();
  test_iter<int*>();
}

constexpr bool test() {
  test_iterators();

  { // Complexity: exactly N evaluations of `gen()` and assignments.
    struct AssignedOnce {
      bool assigned = false;
      constexpr AssignedOnce& operator=(const AssignedOnce&) {
        assert(!assigned);
        assigned = true;
        return *this;
      }
    };

    { // (iterator, sentinel) overload.
      int gen_invocations = 0;
      auto gen = [&gen_invocations] { ++gen_invocations; return AssignedOnce(); };
      constexpr size_t N = 10;
      std::array<AssignedOnce, N> in;

      std::ranges::generate(in.begin(), in.end(), gen);
      assert(std::ranges::all_of(in, &AssignedOnce::assigned));
      assert(gen_invocations == N);
    }

    { // (range) overload.
      int gen_invocations = 0;
      auto gen = [&gen_invocations] { ++gen_invocations; return AssignedOnce(); };
      constexpr size_t N = 10;
      std::array<AssignedOnce, N> in;

      std::ranges::generate(in, gen);
      assert(std::ranges::all_of(in, &AssignedOnce::assigned));
      assert(gen_invocations == N);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
