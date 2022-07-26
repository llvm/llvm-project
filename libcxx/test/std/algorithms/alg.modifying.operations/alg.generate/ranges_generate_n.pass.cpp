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

// template<input_or_output_iterator O, copy_constructible F>
//   requires invocable<F&> && indirectly_writable<O, invoke_result_t<F&>>
//   constexpr O generate_n(O first, iter_difference_t<O> n, F gen);                                // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>

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

// Test type constraints.
// ======================================================

template <class Iter = int*, class Gen = IntGen>
concept HasGenerateNIter =
    requires(Iter&& iter, Gen&& gen) {
      std::ranges::generate_n(std::forward<Iter>(iter), 0, std::forward<Gen>(gen));
    };

static_assert(HasGenerateNIter<int*, IntGen>);

// !input_or_output_iterator<O>
static_assert(!HasGenerateNIter<InputIteratorNotInputOrOutputIterator>);

// !copy_constructible<F>
static_assert(!HasGenerateNIter<int*, UncopyableGen>);

// !invocable<F&>
static_assert(!HasGenerateNIter<int*, UninvocableGen>);

// !indirectly_writable<O, invoke_result_t<F&>>
static_assert(!HasGenerateNIter<int*, IntPtrGen>);

template <class Iter, size_t N, class Gen>
constexpr void test_one(std::array<int, N> in, size_t n, Gen gen, std::array<int, N> expected) {
  assert(n <= N);

  auto begin = Iter(in.data());

  std::same_as<Iter> decltype(auto) result = std::ranges::generate_n(std::move(begin), n, gen);
  assert(base(result) == in.data() + n);
  assert(in == expected);
}

template <class Iter>
constexpr void test_iter() {
  auto gen = [ctr = 1] () mutable { return ctr++; };

  // Empty sequence.
  test_one<Iter, 0>({}, 0, gen, {});
  // 1-element sequence, n = 0.
  test_one<Iter>(std::array{-10}, 0, gen, {-10});
  // 1-element sequence, n = 1.
  test_one<Iter>(std::array{-10}, 1, gen, {1});
  // Longer sequence, n = 3.
  test_one<Iter>(std::array{-10, -20, -30, -40, -50}, 3, gen, {1, 2, 3, -40, -50});
  // Longer sequence, n = 5.
  test_one<Iter>(std::array<int, 5>{}, 5, gen, {1, 2, 3, 4, 5});
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

    int gen_invocations = 0;
    auto gen = [&gen_invocations] { ++gen_invocations; return AssignedOnce(); };
    constexpr size_t N1 = 10;
    constexpr size_t N2 = N1 / 2;
    std::array<AssignedOnce, N1> in;

    auto result = std::ranges::generate_n(in.begin(), N2, gen);
    assert(std::ranges::all_of(std::ranges::subrange(in.begin(), result), &AssignedOnce::assigned));
    assert(std::ranges::none_of(std::ranges::subrange(result, in.end()), &AssignedOnce::assigned));
    assert(gen_invocations == N2);
  }


  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
