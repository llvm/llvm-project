//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<permutable I, sentinel_for<I> S>
//   constexpr subrange<I> ranges::shift_left(I first, S last, iter_difference_t<I> n);

// template<forward_range R>
//   requires permutable<iterator_t<R>>
//   constexpr borrowed_subrange_t<R> ranges::shift_left(R&& r, range_difference_t<R> n)

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>
#include <iterator>

#include "almost_satisfies_types.h"
#include "test_iterators.h"
#include "MoveOnly.h"

template <class Iter, class Sent = Iter>
concept HasShiftLeftIt = requires(Iter iter, Sent sent, std::size_t n) { std::ranges::shift_left(iter, sent, n); };

static_assert(HasShiftLeftIt<int*>);
static_assert(!HasShiftLeftIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasShiftLeftIt<PermutableNotForwardIterator>);
static_assert(!HasShiftLeftIt<PermutableNotSwappable>);

template <class Range>
concept HasShiftLeftR = requires(Range range, std::size_t n) { std::ranges::shift_left(range, n); };

static_assert(HasShiftLeftR<UncheckedRange<int*>>);
static_assert(!HasShiftLeftR<ForwardRangeNotDerivedFrom>);
static_assert(!HasShiftLeftR<PermutableRangeNotForwardIterator>);
static_assert(!HasShiftLeftR<PermutableRangeNotSwappable>);

template <class Iter, class Sent>
constexpr void test_iter_sent() {
  {
    const std::array<int, 8> original = {3, 1, 4, 1, 5, 9, 2, 6};
    std::array<int, 8> scratch;

    // (iterator, sentinel) overload
    for (size_t n = 0; n <= original.size(); ++n) {
      for (size_t k = 0; k <= n + 2; ++k) {
        auto begin = Iter(scratch.data());
        auto end   = Sent(Iter(scratch.data() + n));
        std::ranges::copy(original.begin(), original.begin() + n, begin);
        auto result = std::ranges::shift_left(begin, end, k);

        assert(result.begin() == begin);
        if (k < n) {
          assert(result.end() == Iter(scratch.data() + n - k));
          assert(std::ranges::equal(original.begin() + k, original.begin() + n, result.begin(), result.end()));
        } else {
          assert(result.end() == begin);
          assert(std::ranges::equal(original.begin(), original.begin() + n, begin, end));
        }
      }
    }

    // (range) overload
    for (size_t n = 0; n <= original.size(); ++n) {
      for (size_t k = 0; k <= n + 2; ++k) {
        auto begin = Iter(scratch.data());
        auto end   = Sent(Iter(scratch.data() + n));
        std::ranges::copy(original.begin(), original.begin() + n, begin);
        auto range  = std::ranges::subrange(begin, end);
        auto result = std::ranges::shift_left(range, k);

        assert(result.begin() == begin);
        if (k < n) {
          assert(result.end() == Iter(scratch.data() + n - k));
          assert(std::ranges::equal(original.begin() + k, original.begin() + n, begin, result.end()));
        } else {
          assert(result.end() == begin);
          assert(std::ranges::equal(original.begin(), original.begin() + n, begin, end));
        }
      }
    }
  }

  // n == 0
  {
    std::array<int, 3> input          = {0, 1, 2};
    const std::array<int, 3> expected = {0, 1, 2};

    { // (iterator, sentinel) overload
      auto in     = input;
      auto begin  = Iter(in.data());
      auto end    = Sent(Iter(in.data() + in.size()));
      auto result = std::ranges::shift_left(begin, end, 0);
      assert(std::ranges::equal(expected, result));
      assert(result.begin() == begin);
      assert(result.end() == end);
    }

    { // (range) overload
      auto in     = input;
      auto begin  = Iter(in.data());
      auto end    = Sent(Iter(in.data() + in.size()));
      auto range  = std::ranges::subrange(begin, end);
      auto result = std::ranges::shift_left(range, 0);
      assert(std::ranges::equal(expected, result));
      assert(result.begin() == begin);
      assert(result.end() == end);
    }
  }

  // n == len
  {
    std::array<int, 3> input          = {0, 1, 2};
    const std::array<int, 3> expected = {0, 1, 2};

    { // (iterator, sentinel) overload
      auto in     = input;
      auto begin  = Iter(in.data());
      auto end    = Sent(Iter(in.data() + in.size()));
      auto result = std::ranges::shift_left(begin, end, input.size());
      assert(std::ranges::equal(expected, input));
      assert(result.begin() == begin);
      assert(result.end() == begin);
    }

    { // (range) overload
      auto in     = input;
      auto begin  = Iter(in.data());
      auto end    = Sent(Iter(in.data() + in.size()));
      auto range  = std::ranges::subrange(begin, end);
      auto result = std::ranges::shift_left(range, input.size());
      assert(std::ranges::equal(expected, input));
      assert(result.begin() == begin);
      assert(result.end() == begin);
    }
  }

  // n > len
  {
    std::array<int, 3> input          = {0, 1, 2};
    const std::array<int, 3> expected = {0, 1, 2};

    { // (iterator, sentinel) overload
      auto in     = input;
      auto begin  = Iter(in.data());
      auto end    = Sent(Iter(in.data() + in.size()));
      auto result = std::ranges::shift_left(begin, end, input.size() + 1);
      assert(std::ranges::equal(expected, input));
      assert(result.begin() == begin);
      assert(result.end() == begin);
    }

    { // (range) overload
      auto in     = input;
      auto begin  = Iter(in.data());
      auto end    = Sent(Iter(in.data() + in.size()));
      auto range  = std::ranges::subrange(begin, end);
      auto result = std::ranges::shift_left(range, input.size() + 1);
      assert(std::ranges::equal(expected, input));
      assert(result.begin() == begin);
      assert(result.end() == begin);
    }
  }
}

constexpr bool test() {
  types::for_each(types::forward_iterator_list<int*>{}, []<class Iter> {
    test_iter_sent<Iter, Iter>();
    test_iter_sent<Iter, sentinel_wrapper<Iter>>();
    test_iter_sent<Iter, sized_sentinel<Iter>>();
  });
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
