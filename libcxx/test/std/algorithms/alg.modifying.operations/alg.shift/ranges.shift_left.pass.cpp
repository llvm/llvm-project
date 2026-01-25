//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=9000000

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
#include <vector>

#include "almost_satisfies_types.h"
#include "test_iterators.h"
#include "MoveOnly.h"

struct InvalidDifferenceT {};

template <class Iter, class Sent = Iter, class N = std::iter_difference_t<Iter>>
concept HasShiftLeftIt = requires(Iter iter, Sent sent, N n) { std::ranges::shift_left(iter, sent, n); };

static_assert(HasShiftLeftIt<int*>);
static_assert(HasShiftLeftIt<int*, sentinel_wrapper<int*>>);
static_assert(HasShiftLeftIt<int*, sized_sentinel<int*>>);
static_assert(!HasShiftLeftIt<int*, int*, InvalidDifferenceT>);
static_assert(!HasShiftLeftIt<int*, int, int>);

static_assert(!HasShiftLeftIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasShiftLeftIt<PermutableNotForwardIterator>);
static_assert(!HasShiftLeftIt<PermutableNotSwappable>);

template <class Range, class N = std::ranges::range_difference_t<Range>>
concept HasShiftLeftR = requires(Range range, N n) { std::ranges::shift_left(range, n); };

static_assert(HasShiftLeftR<UncheckedRange<int*>>);
static_assert(!HasShiftLeftR<UncheckedRange<int*>, InvalidDifferenceT>);

static_assert(!HasShiftLeftR<ForwardRangeNotDerivedFrom>);
static_assert(!HasShiftLeftR<PermutableRangeNotForwardIterator>);
static_assert(!HasShiftLeftR<PermutableRangeNotSwappable>);

// An iterator whose iterator_traits::difference_type is not the same as
// std::iter_difference_t<It>.
struct DiffTypeIter {
  int* it_;

  using difference_type   = InvalidDifferenceT;
  using value_type        = int;
  using reference         = int&;
  using pointer           = int*;
  using iterator_category = std::random_access_iterator_tag;

  constexpr DiffTypeIter() : it_() {}
  constexpr explicit DiffTypeIter(int* it) : it_(it) {}

  constexpr reference operator*() const { return *it_; }
  constexpr reference operator[](int n) const { return it_[n]; }

  constexpr DiffTypeIter& operator++() {
    ++it_;
    return *this;
  }
  constexpr DiffTypeIter& operator--() {
    --it_;
    return *this;
  }
  constexpr DiffTypeIter operator++(int) { return DiffTypeIter(it_++); }
  constexpr DiffTypeIter operator--(int) { return DiffTypeIter(it_--); }

  constexpr DiffTypeIter& operator+=(int n) {
    it_ += n;
    return *this;
  }
  constexpr DiffTypeIter& operator-=(int n) {
    it_ -= n;
    return *this;
  }
  friend constexpr DiffTypeIter operator+(DiffTypeIter x, int n) {
    x += n;
    return x;
  }
  friend constexpr DiffTypeIter operator+(int n, DiffTypeIter x) {
    x += n;
    return x;
  }
  friend constexpr DiffTypeIter operator-(DiffTypeIter x, int n) {
    x -= n;
    return x;
  }
  friend constexpr int operator-(DiffTypeIter x, DiffTypeIter y) { return x.it_ - y.it_; }

  friend constexpr bool operator==(const DiffTypeIter& x, const DiffTypeIter& y) { return x.it_ == y.it_; }
  friend constexpr bool operator!=(const DiffTypeIter& x, const DiffTypeIter& y) { return x.it_ != y.it_; }
  friend constexpr bool operator<(const DiffTypeIter& x, const DiffTypeIter& y) { return x.it_ < y.it_; }
  friend constexpr bool operator<=(const DiffTypeIter& x, const DiffTypeIter& y) { return x.it_ <= y.it_; }
  friend constexpr bool operator>(const DiffTypeIter& x, const DiffTypeIter& y) { return x.it_ > y.it_; }
  friend constexpr bool operator>=(const DiffTypeIter& x, const DiffTypeIter& y) { return x.it_ >= y.it_; }
};

template <>
struct std::incrementable_traits<DiffTypeIter> {
  using difference_type = std::ptrdiff_t;
};

struct TrackCopyMove {
  mutable int copy_count = 0;
  int move_count         = 0;

  constexpr TrackCopyMove() = default;
  constexpr TrackCopyMove(const TrackCopyMove& other) : copy_count(other.copy_count), move_count(other.move_count) {
    ++copy_count;
    ++other.copy_count;
  }

  constexpr TrackCopyMove(TrackCopyMove&& other) noexcept : copy_count(other.copy_count), move_count(other.move_count) {
    ++move_count;
    ++other.move_count;
  }
  constexpr TrackCopyMove& operator=(const TrackCopyMove& other) {
    ++copy_count;
    ++other.copy_count;
    return *this;
  }
  constexpr TrackCopyMove& operator=(TrackCopyMove&& other) noexcept {
    ++move_count;
    ++other.move_count;
    return *this;
  }
};

template <class Iter, class Sent>
constexpr void test_iter_sent() {
  {
    const std::array<int, 8> original = {3, 1, 4, 1, 5, 9, 2, 6};

    // (iterator, sentinel) overload
    for (size_t n = 0; n <= original.size(); ++n) {
      for (size_t k = 0; k <= n + 2; ++k) {
        std::array<int, 8> scratch;
        auto begin = Iter(scratch.data());
        auto end   = Sent(Iter(scratch.data() + n));
        std::ranges::copy(original.begin(), original.begin() + n, begin);
        std::same_as<std::ranges::subrange<Iter>> decltype(auto) result = std::ranges::shift_left(begin, end, k);

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
        std::array<int, 8> scratch;
        auto begin = Iter(scratch.data());
        auto end   = Sent(Iter(scratch.data() + n));
        std::ranges::copy(original.begin(), original.begin() + n, begin);
        auto range                                                      = std::ranges::subrange(begin, end);
        std::same_as<std::ranges::subrange<Iter>> decltype(auto) result = std::ranges::shift_left(range, k);

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
      auto in                                                         = input;
      auto begin                                                      = Iter(in.data());
      auto end                                                        = Sent(Iter(in.data() + in.size()));
      std::same_as<std::ranges::subrange<Iter>> decltype(auto) result = std::ranges::shift_left(begin, end, 0);
      assert(std::ranges::equal(expected, result));
      assert(result.begin() == begin);
      assert(result.end() == end);
    }

    { // (range) overload
      auto in                                                         = input;
      auto begin                                                      = Iter(in.data());
      auto end                                                        = Sent(Iter(in.data() + in.size()));
      auto range                                                      = std::ranges::subrange(begin, end);
      std::same_as<std::ranges::subrange<Iter>> decltype(auto) result = std::ranges::shift_left(range, 0);
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
      auto in    = input;
      auto begin = Iter(in.data());
      auto end   = Sent(Iter(in.data() + in.size()));
      std::same_as<std::ranges::subrange<Iter>> decltype(auto) result =
          std::ranges::shift_left(begin, end, input.size());
      assert(std::ranges::equal(expected, input));
      assert(result.begin() == begin);
      assert(result.end() == begin);
    }

    { // (range) overload
      auto in                                                         = input;
      auto begin                                                      = Iter(in.data());
      auto end                                                        = Sent(Iter(in.data() + in.size()));
      auto range                                                      = std::ranges::subrange(begin, end);
      std::same_as<std::ranges::subrange<Iter>> decltype(auto) result = std::ranges::shift_left(range, input.size());
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
      auto in    = input;
      auto begin = Iter(in.data());
      auto end   = Sent(Iter(in.data() + in.size()));
      std::same_as<std::ranges::subrange<Iter>> decltype(auto) result =
          std::ranges::shift_left(begin, end, input.size() + 1);
      assert(std::ranges::equal(expected, input));
      assert(result.begin() == begin);
      assert(result.end() == begin);
    }

    { // (range) overload
      auto in    = input;
      auto begin = Iter(in.data());
      auto end   = Sent(Iter(in.data() + in.size()));
      auto range = std::ranges::subrange(begin, end);
      std::same_as<std::ranges::subrange<Iter>> decltype(auto) result =
          std::ranges::shift_left(range, input.size() + 1);
      assert(std::ranges::equal(expected, input));
      assert(result.begin() == begin);
      assert(result.end() == begin);
    }
  }

  // empty range
  {
    std::vector<int> input          = {};
    const std::vector<int> expected = {};
    { // (iterator, sentinel) overload
      auto in    = input;
      auto begin = Iter(in.data());
      auto end   = Sent(Iter(in.data() + in.size()));
      std::same_as<std::ranges::subrange<Iter>> decltype(auto) result =
          std::ranges::shift_left(begin, end, input.size() + 1);
      assert(std::ranges::equal(expected, input));
      assert(result.begin() == begin);
      assert(result.end() == begin);
    }

    { // (range) overload
      auto in    = input;
      auto begin = Iter(in.data());
      auto end   = Sent(Iter(in.data() + in.size()));
      auto range = std::ranges::subrange(begin, end);
      std::same_as<std::ranges::subrange<Iter>> decltype(auto) result =
          std::ranges::shift_left(range, input.size() + 1);
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
  test_iter_sent<DiffTypeIter, DiffTypeIter>();

  // Complexity: At most (last - first) - n assignments
  {
    constexpr int length = 100;
    constexpr int n      = length / 2;
    auto make_vec        = []() {
      std::vector<TrackCopyMove> vec;
      vec.reserve(length);
      for (int i = 0; i < length; ++i) {
        vec.emplace_back();
      }
      return vec;
    };

    { // (iterator, sentinel) overload
      auto input  = make_vec();
      auto result = std::ranges::shift_left(input.begin(), input.end(), n);
      assert(result.begin() == input.begin());
      assert(result.end() == input.begin() + (length - n));

      auto total_copies = 0;
      auto total_moves  = 0;
      for (auto it = result.begin(); it != result.end(); ++it) {
        const auto& item = *it;
        total_copies += item.copy_count;
        total_moves += item.move_count;
      }

      assert(total_copies == 0);
      assert(total_moves <= length - n);
    }

    { // (range) overload
      auto input  = make_vec();
      auto result = std::ranges::shift_left(input, n);
      assert(result.begin() == input.begin());
      assert(result.end() == input.begin() + (length - n));

      auto total_copies = 0;
      auto total_moves  = 0;
      for (auto it = result.begin(); it != result.end(); ++it) {
        const auto& item = *it;
        total_copies += item.copy_count;
        total_moves += item.move_count;
      }

      assert(total_copies == 0);
      assert(total_moves <= length - n);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
