//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr sentinel(sentinel<!Const> s)
//   requires Const && convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <cassert>
#include <ranges>

#include "../types.h"

struct Sent {
  int i;
  constexpr Sent() = default;
  constexpr Sent(int ii) : i(ii) {}
  friend constexpr bool operator==(int* iter, const Sent& s) { return s.i > *iter; }
};

struct ConstSent {
  int i;
  constexpr ConstSent() = default;
  constexpr ConstSent(int ii) : i(ii) {}
  constexpr ConstSent(const Sent& s) : i(s.i) {}
  friend constexpr bool operator==(int* iter, const ConstSent& s) { return s.i > *iter; }
};

struct Range : std::ranges::view_base {
  int* begin() const;
  Sent end();
  ConstSent end() const;
};

struct Pred {
  bool operator()(int i) const;
};

struct NonConvertConstSent {
  int i;
  constexpr NonConvertConstSent() = default;
  constexpr NonConvertConstSent(int ii) : i(ii) {}
  friend constexpr bool operator==(int* iter, const NonConvertConstSent& s) { return s.i > *iter; }
};

struct NonConvertConstSentRange : std::ranges::view_base {
  int* begin() const;
  Sent end();
  NonConvertConstSent end() const;
};

// Test Constraint
static_assert(std::is_constructible_v<std::ranges::sentinel_t<const std::ranges::take_while_view<Range, Pred>>,
                                      std::ranges::sentinel_t<std::ranges::take_while_view<Range, Pred>>>);

// !Const
static_assert(!std::is_constructible_v<std::ranges::sentinel_t<std::ranges::take_while_view<Range, Pred>>,
                                       std::ranges::sentinel_t<const std::ranges::take_while_view<Range, Pred>>>);

// !convertible_to<sentinel_t<V>, sentinel_t<Base>>
static_assert(!std::is_constructible_v<
              std::ranges::sentinel_t<const std::ranges::take_while_view<NonConvertConstSentRange, Pred>>,
              std::ranges::sentinel_t<std::ranges::take_while_view<NonConvertConstSentRange, Pred>>>);

constexpr bool test() {
  // base is init correctly
  {
    using R             = std::ranges::take_while_view<Range, bool (*)(int)>;
    using Sentinel      = std::ranges::sentinel_t<R>;
    using ConstSentinel = std::ranges::sentinel_t<const R>;
    static_assert(!std::same_as<Sentinel, ConstSentinel>);

    Sentinel s1(Sent{5}, nullptr);
    ConstSentinel s2 = s1;
    assert(s2.base().i == 5);
  }

  // pred is init correctly
  {
    bool called = false;
    auto pred   = [&](int) {
      called = true;
      return false;
    };

    using R             = std::ranges::take_while_view<Range, decltype(pred)>;
    using Sentinel      = std::ranges::sentinel_t<R>;
    using ConstSentinel = std::ranges::sentinel_t<const R>;
    static_assert(!std::same_as<Sentinel, ConstSentinel>);

    int i     = 10;
    int* iter = &i;
    Sentinel s1(Sent{0}, &pred);
    ConstSentinel s2 = s1;

    [[maybe_unused]] bool b = iter == s2;
    assert(called);
  }

  // LWG 3708 `take_while_view::sentinel`'s conversion constructor should move
  {
    struct MoveOnlyConvert {
      int i;
      constexpr MoveOnlyConvert() = default;
      constexpr MoveOnlyConvert(Sent&& s) : i(s.i) { s.i = 0; }
      constexpr bool operator==(int* iter) const { return i > *iter; }
    };

    struct Rng : std::ranges::view_base {
      int* begin() const;
      Sent end();
      MoveOnlyConvert end() const;
    };

    using R             = std::ranges::take_while_view<Rng, Pred>;
    using Sentinel      = std::ranges::sentinel_t<R>;
    using ConstSentinel = std::ranges::sentinel_t<const R>;
    static_assert(!std::same_as<Sentinel, ConstSentinel>);

    Sentinel s1(Sent{5}, nullptr);
    ConstSentinel s2 = s1;
    assert(s2.base().i == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
