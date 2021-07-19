//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// template<class I>
// unspecified iter_move;

#include <iterator>

#include <array>
#include <algorithm>
#include <cassert>
#include <utility>

#include "../unqualified_lookup_wrapper.h"

// Wrapper around an iterator for testing `iter_move` when an unqualified call to `iter_move` isn't
// possible.
template <typename I>
class iterator_wrapper {
public:
  iterator_wrapper() = default;

  constexpr explicit iterator_wrapper(I i) noexcept : base_(std::move(i)) {}

  // `noexcept(false)` is used to check that this operator is called.
  constexpr decltype(auto) operator*() const& noexcept(false) { return *base_; }

  // `noexcept` is used to check that this operator is called.
  constexpr auto&& operator*() && noexcept { return std::move(*base_); }

  constexpr iterator_wrapper& operator++() noexcept {
    ++base_;
    return *this;
  }

  constexpr void operator++(int) noexcept { ++base_; }

  constexpr bool operator==(iterator_wrapper const& other) const noexcept { return base_ == other.base_; }

private:
  I base_ = I{};
};

template <typename I>
constexpr void unqualified_lookup_move(I first_, I last_, I result_first_, I result_last_) {
  auto first = ::check_unqualified_lookup::unqualified_lookup_wrapper{std::move(first_)};
  auto last = ::check_unqualified_lookup::unqualified_lookup_wrapper{std::move(last_)};
  auto result_first = ::check_unqualified_lookup::unqualified_lookup_wrapper{std::move(result_first_)};
  auto result_last = ::check_unqualified_lookup::unqualified_lookup_wrapper{std::move(result_last_)};

  static_assert(!noexcept(std::ranges::iter_move(first)), "unqualified-lookup case not being chosen");

  for (; first != last && result_first != result_last; (void)++first, ++result_first) {
    *result_first = std::ranges::iter_move(first);
  }
}

template <typename I>
constexpr void lvalue_move(I first_, I last_, I result_first_, I result_last_) {
  auto first = iterator_wrapper{std::move(first_)};
  auto last = ::iterator_wrapper{std::move(last_)};
  auto result_first = iterator_wrapper{std::move(result_first_)};
  auto result_last = iterator_wrapper{std::move(result_last_)};

  static_assert(!noexcept(std::ranges::iter_move(first)), "`operator*() const&` is not noexcept, and there's no hidden "
                                                          "friend iter_move.");

  for (; first != last && result_first != result_last; (void)++first, ++result_first) {
    *result_first = std::ranges::iter_move(first);
  }
}

template <typename I>
constexpr void rvalue_move(I first_, I last_, I result_first_, I result_last_) {
  auto first = iterator_wrapper{std::move(first_)};
  auto last = iterator_wrapper{std::move(last_)};
  auto result_first = iterator_wrapper{std::move(result_first_)};
  auto result_last = iterator_wrapper{std::move(result_last_)};

  static_assert(noexcept(std::ranges::iter_move(std::move(first))),
                "`operator*() &&` is noexcept, and there's no hidden friend iter_move.");

  for (; first != last && result_first != result_last; (void)++first, ++result_first) {
    auto i = first;
    *result_first = std::ranges::iter_move(std::move(i));
  }
}

template <bool NoExcept>
struct WithADL {
  WithADL() = default;
  constexpr int operator*() const { return 0; }
  constexpr WithADL& operator++();
  constexpr void operator++(int);
  constexpr bool operator==(WithADL const&) const;
  friend constexpr int iter_move(WithADL&&) noexcept(NoExcept) { return 0; }
};

template <bool NoExcept>
struct WithoutADL {
  WithoutADL() = default;
  constexpr int operator*() const noexcept(NoExcept) { return 0; }
  constexpr WithoutADL& operator++();
  constexpr void operator++(int);
  constexpr bool operator==(WithoutADL const&) const;
};

constexpr bool check_iter_move() {
  constexpr int full_size = 100;
  constexpr int half_size = full_size / 2;
  constexpr int reset = 0;
  auto v1 = std::array<move_tracker, full_size>{};

  auto move_counter_is = [](auto const n) { return [n](auto const& x) { return x.moves() == n; }; };

  auto v2 = std::array<move_tracker, half_size>{};
  unqualified_lookup_move(v1.begin(), v1.end(), v2.begin(), v2.end());
  assert(std::all_of(v1.cbegin(), v1.cend(), move_counter_is(reset)));
  assert(std::all_of(v2.cbegin(), v2.cend(), move_counter_is(1)));

  auto v3 = std::array<move_tracker, half_size>{};
  unqualified_lookup_move(v1.begin() + half_size, v1.end(), v3.begin(), v3.end());
  assert(std::all_of(v1.cbegin(), v1.cend(), move_counter_is(reset)));
  assert(std::all_of(v3.cbegin(), v3.cend(), move_counter_is(1)));

  auto v4 = std::array<move_tracker, half_size>{};
  unqualified_lookup_move(v3.begin(), v3.end(), v4.begin(), v4.end());
  assert(std::all_of(v3.cbegin(), v3.cend(), move_counter_is(reset)));
  assert(std::all_of(v4.cbegin(), v4.cend(), move_counter_is(2)));

  lvalue_move(v2.begin(), v2.end(), v1.begin() + half_size, v1.end());
  assert(std::all_of(v2.cbegin(), v2.cend(), move_counter_is(reset)));
  assert(std::all_of(v1.cbegin() + half_size, v1.cend(), move_counter_is(2)));

  lvalue_move(v4.begin(), v4.end(), v1.begin(), v1.end());
  assert(std::all_of(v4.cbegin(), v4.cend(), move_counter_is(reset)));
  assert(std::all_of(v1.cbegin(), v1.cbegin() + half_size, move_counter_is(3)));

  rvalue_move(v1.begin(), v1.end(), v2.begin(), v2.end());
  assert(std::all_of(v1.cbegin(), v1.cbegin() + half_size, move_counter_is(reset)));
  assert(std::all_of(v2.cbegin(), v2.cend(), move_counter_is(4)));

  rvalue_move(v1.begin() + half_size, v1.end(), v3.begin(), v3.end());
  assert(std::all_of(v1.cbegin(), v1.cend(), move_counter_is(reset)));
  assert(std::all_of(v3.cbegin(), v3.cend(), move_counter_is(3)));

  auto unscoped = check_unqualified_lookup::unscoped_enum::a;
  assert(std::ranges::iter_move(unscoped) == check_unqualified_lookup::unscoped_enum::a);
  assert(!noexcept(std::ranges::iter_move(unscoped)));

  auto scoped = check_unqualified_lookup::scoped_enum::a;
  assert(std::ranges::iter_move(scoped) == nullptr);
  assert(noexcept(std::ranges::iter_move(scoped)));

  auto some_union = check_unqualified_lookup::some_union{0};
  assert(std::ranges::iter_move(some_union) == 0);
  assert(!noexcept(std::ranges::iter_move(some_union)));

  // Check noexcept-correctness
  static_assert(noexcept(std::ranges::iter_move(std::declval<WithADL<true>>())));
  static_assert(!noexcept(std::ranges::iter_move(std::declval<WithADL<false>>())));
  static_assert(noexcept(std::ranges::iter_move(std::declval<WithoutADL<true>>())));
  static_assert(!noexcept(std::ranges::iter_move(std::declval<WithoutADL<false>>())));

  return true;
}

template <typename T>
concept can_iter_move = requires (T t) { std::ranges::iter_move(t); };

int main(int, char**) {
  static_assert(check_iter_move());
  check_iter_move();

  // Make sure that `iter_move` SFINAEs away when the type can't be iter_move'd
  {
    struct NoIterMove { };
    static_assert(!can_iter_move<NoIterMove>);
  }

  return 0;
}
