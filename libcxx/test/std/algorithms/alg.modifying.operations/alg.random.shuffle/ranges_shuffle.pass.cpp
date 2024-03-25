//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<random_access_iterator I, sentinel_for<I> S, class Gen>
//   requires permutable<I> &&
//            uniform_random_bit_generator<remove_reference_t<Gen>>
//   I shuffle(I first, S last, Gen&& g);                                                           // Since C++20
//
// template<random_access_range R, class Gen>
//   requires permutable<iterator_t<R>> &&
//            uniform_random_bit_generator<remove_reference_t<Gen>>
//   borrowed_iterator_t<R> shuffle(R&& r, Gen&& g);                                                // Since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <random>
#include <ranges>
#include <utility>

#include "almost_satisfies_types.h"
#include "test_iterators.h"
#include "test_macros.h"

class RandGen {
public:
  constexpr static std::size_t min() { return 0; }
  constexpr static std::size_t max() { return 255; }

  constexpr std::size_t operator()() {
    flip = !flip;
    return flip;
  }

private:
  bool flip = false;
};

static_assert(std::uniform_random_bit_generator<RandGen>);
// `std::uniform_random_bit_generator` is a subset of requirements of `__libcpp_random_is_valid_urng`. Make sure that
// a type satisfying the required minimum is still accepted by `ranges::shuffle`.
LIBCPP_STATIC_ASSERT(!std::__libcpp_random_is_valid_urng<RandGen>::value);

struct BadGen {
  constexpr static std::size_t min() { return 255; }
  constexpr static std::size_t max() { return 0; }
  constexpr std::size_t operator()() const;
};
static_assert(!std::uniform_random_bit_generator<BadGen>);

// Test constraints of the (iterator, sentinel) overload.
// ======================================================

template <class Iter = int*, class Sent = int*, class Gen = RandGen>
concept HasShuffleIter =
    requires(Iter&& iter, Sent&& sent, Gen&& gen) {
      std::ranges::shuffle(std::forward<Iter>(iter), std::forward<Sent>(sent), std::forward<Gen>(gen));
    };

static_assert(HasShuffleIter<int*, int*, RandGen>);

// !random_access_iterator<I>
static_assert(!HasShuffleIter<RandomAccessIteratorNotDerivedFrom>);
static_assert(!HasShuffleIter<RandomAccessIteratorBadIndex>);

// !sentinel_for<S, I>
static_assert(!HasShuffleIter<int*, SentinelForNotSemiregular>);
static_assert(!HasShuffleIter<int*, SentinelForNotWeaklyEqualityComparableWith>);

// !permutable<I>
static_assert(!HasShuffleIter<PermutableNotForwardIterator>);
static_assert(!HasShuffleIter<PermutableNotSwappable>);

// !uniform_random_bit_generator<remove_reference_t<Gen>>
static_assert(!HasShuffleIter<int*, int*, BadGen>);

// Test constraints of the (range) overload.
// =========================================

template <class Range, class Gen = RandGen>
concept HasShuffleRange =
    requires(Range&& range, Gen&& gen) {
      std::ranges::shuffle(std::forward<Range>(range), std::forward<Gen>(gen));
    };

template <class T>
using R = UncheckedRange<T>;

static_assert(HasShuffleRange<R<int*>, RandGen>);

// !random_access_range<R>
static_assert(!HasShuffleRange<RandomAccessRangeNotDerivedFrom>);
static_assert(!HasShuffleRange<RandomAccessRangeBadIndex>);

// !permutable<iterator_t<R>>
static_assert(!HasShuffleRange<PermutableNotForwardIterator>);
static_assert(!HasShuffleRange<PermutableNotSwappable>);

// !uniform_random_bit_generator<remove_reference_t<Gen>>
static_assert(!HasShuffleRange<R<int*>, BadGen>);

template <class Iter, class Sent, std::size_t N, class Gen>
void test_one(const std::array<int, N> input, Gen gen) {
  { // (iterator, sentinel) overload.
    auto shuffled = input;
    auto begin = Iter(shuffled.data());
    auto end = Sent(Iter(shuffled.data() + shuffled.size()));

    std::same_as<Iter> decltype(auto) result = std::ranges::shuffle(begin, end, gen);

    assert(result == Iter(shuffled.data() + shuffled.size()));
    // TODO(ranges): uncomment `ranges::is_permutation` once https://reviews.llvm.org/D127194 lands and remove sorting.
    //assert(std::ranges::is_permutation(shuffled, input);
    {
      auto shuffled_sorted = shuffled;
      std::ranges::sort(shuffled_sorted);
      auto original_sorted = input;
      std::ranges::sort(original_sorted);
      assert(shuffled_sorted == original_sorted);
    }
  }

  { // (range) overload.
    auto shuffled = input;
    auto begin = Iter(shuffled.data());
    auto end = Sent(Iter(shuffled.data() + shuffled.size()));
    auto range = std::ranges::subrange(begin, end);

    std::same_as<Iter> decltype(auto) result = std::ranges::shuffle(range, gen);

    assert(result == Iter(shuffled.data() + shuffled.size()));
    // TODO(ranges): uncomment `ranges::is_permutation` once https://reviews.llvm.org/D127194 lands and remove sorting.
    //assert(std::ranges::is_permutation(shuffled, input);
    {
      auto shuffled_sorted = shuffled;
      std::ranges::sort(shuffled_sorted);
      auto original_sorted = input;
      std::ranges::sort(original_sorted);
      assert(shuffled_sorted == original_sorted);
    }
  }
}

template <class Iter, class Sent>
void test_iterators_iter_sent() {
  RandGen gen;

  // Empty sequence.
  test_one<Iter, Sent, 0>({}, gen);
  // 1-element sequence.
  test_one<Iter, Sent, 1>({1}, gen);
  // 2-element sequence.
  test_one<Iter, Sent, 2>({2, 1}, gen);
  // 3-element sequence.
  test_one<Iter, Sent, 3>({2, 1, 3}, gen);
  // Longer sequence.
  test_one<Iter, Sent, 8>({2, 1, 3, 6, 8, 4, 11, 5}, gen);
  // Longer sequence with duplicates.
  test_one<Iter, Sent, 8>({2, 1, 3, 6, 2, 8, 1, 6}, gen);
  // All elements are the same.
  test_one<Iter, Sent, 3>({1, 1, 1}, gen);
}

template <class Iter>
void test_iterators_iter() {
  test_iterators_iter_sent<Iter, Iter>();
  test_iterators_iter_sent<Iter, sentinel_wrapper<Iter>>();
}

void test_iterators() {
  test_iterators_iter<random_access_iterator<int*>>();
  test_iterators_iter<contiguous_iterator<int*>>();
  test_iterators_iter<int*>();
}

// Checks the logic for wrapping the given iterator to make sure it works correctly regardless of the value category of
// the given generator object.
template <class Gen, bool CheckConst = true>
void test_generator() {
  std::array in = {1, 2, 3, 4, 5, 6, 7, 8};
  auto begin = in.begin();
  auto end = in.end();

  { // Lvalue.
    Gen g;
    std::ranges::shuffle(begin, end, g);
    std::ranges::shuffle(in, g);
  }

  if constexpr (CheckConst) { // Const lvalue.
    const Gen g;
    std::ranges::shuffle(begin, end, g);
    std::ranges::shuffle(in, g);
  }

  { // Prvalue.
    std::ranges::shuffle(begin, end, Gen());
    std::ranges::shuffle(in, Gen());
  }

  { // Xvalue.
    Gen g1, g2;
    std::ranges::shuffle(begin, end, std::move(g1));
    std::ranges::shuffle(in, std::move(g2));
  }
}

// Checks the logic for wrapping the given iterator to make sure it works correctly regardless of whether the given
// generator class has a const or non-const invocation operator (or both).
void test_generators() {
  struct GenBase {
    constexpr static std::size_t min() { return 0; }
    constexpr static std::size_t max() { return 255; }
  };
  struct NonconstGen : GenBase {
    std::size_t operator()() { return 1; }
  };
  struct ConstGen : GenBase {
    std::size_t operator()() const { return 1; }
  };
  struct ConstAndNonconstGen : GenBase {
    std::size_t operator()() { return 1; }
    std::size_t operator()() const { return 1; }
  };

  test_generator<ConstGen>();
  test_generator<NonconstGen, /*CheckConst=*/false>();
  test_generator<ConstAndNonconstGen>();
}

void test() {
  test_iterators();
  test_generators();

  { // Complexity: Exactly `(last - first) - 1` swaps.
    {
      std::array in = {-2, -5, -8, -11, -10, -5, 1, 3, 9, 6, 8, 2, 4, 2}; //14

      int swaps = 0;
      auto begin = adl::Iterator::TrackSwaps(in.data(), swaps);
      auto end = adl::Iterator::TrackSwaps(in.data() + in.size(), swaps);

      std::ranges::shuffle(begin, end, RandGen());
      int expected = in.size() - 1;
      // Note: our implementation doesn't perform a swap when the distribution returns 0, so the actual number of swaps
      // might be less than specified in the standard.
      assert(swaps <= expected);
      swaps = 0;
    }
  }
}

int main(int, char**) {
  test();
  // Note: `ranges::shuffle` is not `constexpr`.

  return 0;
}
