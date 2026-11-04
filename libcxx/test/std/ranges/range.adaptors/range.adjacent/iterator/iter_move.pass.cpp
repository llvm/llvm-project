//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// friend constexpr auto iter_move(const iterator& i) noexcept(see below);

#include <array>
#include <cassert>
#include <iterator>
#include <ranges>
#include <tuple>

#include "../helpers.h"
#include "../../range_adaptor_types.h"

struct ThrowingMove {
  ThrowingMove() = default;
  ThrowingMove(ThrowingMove&&) {}
};

class IterMoveMayThrowIter {
  int* it_;

public:
  using value_type      = int;
  using difference_type = typename std::iterator_traits<int*>::difference_type;

  constexpr IterMoveMayThrowIter() = default;
  explicit constexpr IterMoveMayThrowIter(int* it) : it_(it) {}

  friend constexpr decltype(auto) iter_move(const IterMoveMayThrowIter& it) noexcept(false) {
    return std::ranges::iter_move(it.it_);
  }

  friend constexpr bool operator==(const IterMoveMayThrowIter& x, const IterMoveMayThrowIter& y) {
    return x.it_ == y.it_;
  }

  constexpr decltype(auto) operator*() const { return *it_; }
  constexpr IterMoveMayThrowIter& operator++() {
    ++it_;
    return *this;
  }
  constexpr IterMoveMayThrowIter operator++(int) {
    auto tmp(*this);
    ++(*this);
    return tmp;
  }
};

class IterMoveMayThrowRange {
  int* buffer_;
  std::size_t size_;

public:
  constexpr IterMoveMayThrowRange(int* buffer, std::size_t size) : buffer_(buffer), size_(size) {}
  constexpr IterMoveMayThrowIter begin() const { return IterMoveMayThrowIter{buffer_}; }
  constexpr IterMoveMayThrowIter end() const { return IterMoveMayThrowIter{buffer_ + size_}; }
};

template <std::size_t N>
constexpr void test() {
  {
    // underlying iter_move noexcept
    int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto v = buffer | std::views::adjacent<N>;

    auto it = v.begin();
    static_assert(noexcept(std::ranges::iter_move(it)));

    std::same_as<expectedTupleType<N, int&&>> decltype(auto) res = std::ranges::iter_move(it);

    assert(&std::get<0>(res) == &buffer[0]);
    if constexpr (N >= 2)
      assert(&std::get<1>(res) == &buffer[1]);
    if constexpr (N >= 3)
      assert(&std::get<2>(res) == &buffer[2]);
    if constexpr (N >= 4)
      assert(&std::get<3>(res) == &buffer[3]);
    if constexpr (N >= 5)
      assert(&std::get<4>(res) == &buffer[4]);
  }

  {
    // underlying iter_move may throw
    int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto v = IterMoveMayThrowRange{buffer, 9} | std::views::adjacent<N>;

    auto it = v.begin();
    static_assert(!noexcept(std::ranges::iter_move(it)));

    std::same_as<expectedTupleType<N, int&&>> decltype(auto) res = std::ranges::iter_move(it);
    assert(&std::get<0>(res) == &buffer[0]);
    if constexpr (N >= 2)
      assert(&std::get<1>(res) == &buffer[1]);
    if constexpr (N >= 3)
      assert(&std::get<2>(res) == &buffer[2]);
    if constexpr (N >= 4)
      assert(&std::get<3>(res) == &buffer[3]);
    if constexpr (N >= 5)
      assert(&std::get<4>(res) == &buffer[4]);
  }

  {
    // !is_nothrow_move_constructible_v<range_rvalue_reference_t<Base>>
    // underlying iter_move may throw
    auto throwingMoveRange =
        std::views::iota(0, 9) | std::views::transform([](auto) noexcept { return ThrowingMove{}; });
    auto v  = throwingMoveRange | std::views::adjacent<N>;
    auto it = v.begin();
    static_assert(!noexcept(std::ranges::iter_move(it)));
  }

  {
    // underlying iterators' iter_move are called through ranges::iter_move
    auto rng = adltest::IterMoveSwapRange{};
    auto v   = rng | std::views::adjacent<N>;
    assert(rng.iter_move_called_times == 0);
    auto it = v.begin();
    {
      [[maybe_unused]] auto&& i = std::ranges::iter_move(it);
      assert(rng.iter_move_called_times == N);
    }
    {
      [[maybe_unused]] auto&& i = std::ranges::iter_move(it);
      assert(rng.iter_move_called_times == 2 * N);
    }
  }
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<4>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
