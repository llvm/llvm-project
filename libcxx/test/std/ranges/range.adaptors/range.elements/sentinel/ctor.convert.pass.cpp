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
#include <tuple>

#include "../types.h"

struct Sent {
  int i;
  constexpr Sent() = default;
  constexpr Sent(int ii) : i(ii) {}
  friend constexpr bool operator==(std::tuple<int>*, const Sent&) { return true; }
};

struct ConstSent {
  int i;
  constexpr ConstSent() = default;
  constexpr ConstSent(int ii) : i(ii) {}
  constexpr ConstSent(const Sent& s) : i(s.i) {}
  friend constexpr bool operator==(std::tuple<int>*, const ConstSent&) { return true; }
};

struct Range : std::ranges::view_base {
  std::tuple<int>* begin() const;
  Sent end();
  ConstSent end() const;
};

struct NonConvertConstSent {
  int i;
  constexpr NonConvertConstSent() = default;
  constexpr NonConvertConstSent(int ii) : i(ii) {}
  friend constexpr bool operator==(std::tuple<int>*, const NonConvertConstSent&) { return true; }
};

struct NonConvertConstSentRange : std::ranges::view_base {
  std::tuple<int>* begin() const;
  Sent end();
  NonConvertConstSent end() const;
};

// Test Constraint
static_assert(std::is_constructible_v<std::ranges::sentinel_t<const std::ranges::elements_view<Range, 0>>,
                                      std::ranges::sentinel_t<std::ranges::elements_view<Range, 0>>>);

// !Const
static_assert(!std::is_constructible_v<std::ranges::sentinel_t<std::ranges::elements_view<Range, 0>>,
                                       std::ranges::sentinel_t<const std::ranges::elements_view<Range, 0>>>);

// !convertible_to<sentinel_t<V>, sentinel_t<Base>>
static_assert(!std::is_constructible_v<
              std::ranges::sentinel_t<const std::ranges::elements_view<NonConvertConstSentRange, 0>>,
              std::ranges::sentinel_t<std::ranges::elements_view<NonConvertConstSentRange, 0>>>);

constexpr bool test() {
  // base is init correctly
  {
    using R             = std::ranges::elements_view<Range, 0>;
    using Sentinel      = std::ranges::sentinel_t<R>;
    using ConstSentinel = std::ranges::sentinel_t<const R>;
    static_assert(!std::same_as<Sentinel, ConstSentinel>);

    Sentinel s1(Sent{5});
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
