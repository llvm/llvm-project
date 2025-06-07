//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<input_iterator I, sentinel_for<I> S,
//          indirectly-binary-left-foldable<T, I> F>
//   requires constructible_from<iter_value_t<I>, iter_reference_t<I>>
//   constexpr see below fold_left_first_with_iter(I first, S last, F f);                       // since C++23

// template<input_range R, indirectly-binary-left-foldable<T, iterator_t<R>> F>
//   requires constructible_from<range_value_t<R>, range_reference_t<R>>
//   constexpr see below fold_left_first_with_iter(R&& r, F f);                                 // since C++23

// REQUIRES: std-at-least-c++23

// MSVC warning C4244: 'argument': conversion from 'double' to 'const int', possible loss of data
// ADDITIONAL_COMPILE_FLAGS(cl-style-warnings): /wd4244

#include <algorithm>
#include <cassert>
#include <concepts>
#include <functional>
#include <iterator>
#include <ranges>
#include <string_view>
#include <string>
#include <vector>
#include <optional>

#include "test_macros.h"
#include "test_range.h"
#include "invocable_with_telemetry.h"
#include "maths.h"

template <class Result, class Range, class T>
concept is_in_value_result =
    std::same_as<Result, std::ranges::fold_left_first_with_iter_result<std::ranges::iterator_t<Range>, T>>;

struct Integer {
  int value;

  constexpr Integer(int const x) : value(x) {}

  constexpr Integer plus(int const x) const { return Integer{value + x}; }

  friend constexpr bool operator==(Integer const& x, Integer const& y) = default;
};

template <std::ranges::input_range R, class F, std::equality_comparable Expected>
  requires std::copyable<R>
constexpr void check_iterator(R& r, F f, std::optional<Expected> const& expected) {
  {
    is_in_value_result<R, std::optional<Expected>> decltype(auto) result =
        std::ranges::fold_left_first_with_iter(r.begin(), r.end(), f);
    assert(result.in == r.end());
    assert(result.value == expected);
  }

  {
    is_in_value_result<R, std::optional<Expected>> decltype(auto) result = std::ranges::fold_left_first_with_iter(r, f);
    assert(result.in == r.end());
    assert(result.value == expected);
  }

  {
    auto telemetry = invocable_telemetry();
    auto f2        = invocable_with_telemetry(f, telemetry);
    is_in_value_result<R, std::optional<Expected>> decltype(auto) result =
        std::ranges::fold_left_first_with_iter(r.begin(), r.end(), f2);
    assert(result.in == r.end());
    assert(result.value == expected);
    if (result.value.has_value()) {
      assert(telemetry.invocations == std::ranges::distance(r) - 1);
      assert(telemetry.moves == 0);
      assert(telemetry.copies == 1);
    }
  }

  {
    auto telemetry = invocable_telemetry();
    auto f2        = invocable_with_telemetry(f, telemetry);
    is_in_value_result<R, std::optional<Expected>> decltype(auto) result =
        std::ranges::fold_left_first_with_iter(r, f2);
    assert(result.in == r.end());
    assert(result.value == expected);
    if (result.value.has_value()) {
      assert(telemetry.invocations == std::ranges::distance(r) - 1);
      assert(telemetry.moves == 0);
      assert(telemetry.copies == 1);
    }
  }
}

template <std::ranges::input_range R, class F, std::equality_comparable Expected>
  requires std::copyable<R>
constexpr void check(R r, F f, std::optional<Expected> const& expected) {
  check_iterator(r, f, expected);
}

constexpr void empty_range_test_case() {
  auto const data = std::vector<int>{};
  check(data, std::plus(), std::optional<int>());
  check(data | std::views::take_while([](auto) { return false; }), std::plus(), std::optional<int>());
}

constexpr void common_range_test_case() {
  auto const data = std::vector<int>{1, 2, 3, 4};
  check(data, std::plus(), std::optional(triangular_sum(data)));
  check(data, std::multiplies(), std::optional(factorial(data.back())));

  auto multiply_with_prev = [n = 1](auto const x, auto const y) mutable {
    auto const result = x * y * n;
    n                 = y;
    return static_cast<std::size_t>(result);
  };
  check(data, multiply_with_prev, std::optional(factorial(data.size()) * factorial(data.size() - 1)));
}

constexpr void non_common_range_test_case() {
  auto parse = [](std::string_view const s) {
    return s == "zero"  ? 0.0
         : s == "one"   ? 1.0
         : s == "two"   ? 2.0
         : s == "three" ? 3.0
         : s == "four"  ? 4.0
         : s == "five"  ? 5.0
         : s == "six"   ? 6.0
         : s == "seven" ? 7.0
         : s == "eight" ? 8.0
         : s == "nine"  ? 9.0
                        : (assert(false), 10.0); // the number here is arbitrary
  };

  {
    auto data  = std::vector<std::string>{"five", "three", "two", "six", "one", "four"};
    auto range = data | std::views::transform(parse);
    check(range, std::plus(), std::optional(triangular_sum(range)));
  }

  {
    auto data           = std::string("five three two six one four");
    auto to_string_view = [](auto&& r) {
      auto const n = std::ranges::distance(r);
      return std::string_view(&*r.begin(), n);
    };
    auto range =
        std::views::lazy_split(data, ' ') | std::views::transform(to_string_view) | std::views::transform(parse);
    check(range, std::plus(), std::optional(triangular_sum(range)));
  }
}

constexpr bool test_case() {
  empty_range_test_case();
  common_range_test_case();
  non_common_range_test_case();
  return true;
}

int main(int, char**) {
  test_case();
  static_assert(test_case());
  return 0;
}
