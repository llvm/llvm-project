//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: std-at-least-c++23

// template<bidirectional_iterator I, sentinel_for<I> S,
//         indirectly-binary-right-foldable<iter_value_t<I>, I> F>
//   requires constructible_from<iter_value_t<I>, iter_reference_t<I>>
//   constexpr auto ranges::fold_right_last(I first, S last, F f);

// template<bidirectional_range R,
//         indirectly-binary-right-foldable<range_value_t<R>, iterator_t<R>> F>
//   requires constructible_from<range_value_t<R>, range_reference_t<R>>
//   constexpr auto ranges::fold_right_last(R&& r, F f);

#include <algorithm>
#include <cassert>
#include <concepts>
#include <deque>
#include <functional>
#include <iterator>
#include <list>
#include <optional>
#include <ranges>
#include <set>
#include <string_view>
#include <string>
#include <vector>

#include "test_macros.h"
#include "test_range.h"
#include "invocable_with_telemetry.h"
#include "maths.h"

using std::ranges::fold_right_last;

template <std::ranges::input_range R, class F, std::equality_comparable Expected>
  requires std::copyable<R>
constexpr void check_iterator(R& r, F f, std::optional<Expected> const& expected) {
  {
    std::same_as<std::optional<Expected>> decltype(auto) result = fold_right_last(r.begin(), r.end(), f);
    assert(result == expected);
  }

  {
    auto telemetry                                              = invocable_telemetry();
    auto f2                                                     = invocable_with_telemetry(f, telemetry);
    std::same_as<std::optional<Expected>> decltype(auto) result = fold_right_last(r.begin(), r.end(), f2);
    assert(result == expected);
    if (expected.has_value()) {
      assert(telemetry.invocations == std::ranges::distance(r) - 1);
      assert(telemetry.moves == 1);
      assert(telemetry.copies == 1);
    }
  }
}

template <std::ranges::input_range R, class F, std::equality_comparable Expected>
  requires std::copyable<R>
constexpr void check_lvalue_range(R& r, F f, std::optional<Expected> const& expected) {
  {
    std::same_as<std::optional<Expected>> decltype(auto) result = fold_right_last(r, f);
    assert(result == expected);
  }

  {
    auto telemetry                                              = invocable_telemetry();
    auto f2                                                     = invocable_with_telemetry(f, telemetry);
    std::same_as<std::optional<Expected>> decltype(auto) result = fold_right_last(r, f2);
    assert(result == expected);
    if (expected.has_value()) {
      assert(telemetry.invocations == std::ranges::distance(r) - 1);
      assert(telemetry.moves == 0);
      assert(telemetry.copies == 1);
    }
  }
}

template <std::ranges::input_range R, class F, std::equality_comparable Expected>
  requires std::copyable<R>
constexpr void check_rvalue_range(R& r, F f, std::optional<Expected> const& expected) {
  {
    auto r2                                                     = r;
    std::same_as<std::optional<Expected>> decltype(auto) result = fold_right_last(std::move(r2), f);
    assert(result == expected);
  }

  {
    auto telemetry                                              = invocable_telemetry();
    auto f2                                                     = invocable_with_telemetry(f, telemetry);
    auto r2                                                     = r;
    std::same_as<std::optional<Expected>> decltype(auto) result = fold_right_last(std::move(r2), f2);
    assert(result == expected);
    if (expected.has_value()) {
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
  check_lvalue_range(r, f, expected);
  check_rvalue_range(r, f, expected);
}

constexpr void empty_range_test_case() {
  auto const data = std::vector<int>{};
  check(data, std::plus(), std::optional<int>());
  check(data, std::multiplies(), std::optional<int>());
}

constexpr void common_range_test_case() {
  auto const data = std::vector<int>{1, 2, 3, 4};
  check(data, std::plus(), std::optional(triangular_sum(data)));
  check(data, std::multiplies(), std::optional(factorial(data.back())));

  auto multiply_with_next = [n = 1](auto const x, auto const y) mutable {
    auto const result = x * y * n;
    n                 = x;
    return static_cast<std::size_t>(result);
  };
  check(data, multiply_with_next, std::optional(factorial(data.size()) * factorial(data.size() - 1)));

  auto fib = [n = 1](auto x, auto) mutable {
    auto old_x = x;
    x += n;
    n = old_x;
    return x;
  };
  check(data, fib, std::optional(fibonacci(data.back())));
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
}

constexpr bool test_case() {
  empty_range_test_case();
  common_range_test_case();
  non_common_range_test_case();
  return true;
}

// Most containers aren't constexpr
void runtime_only_test_case() {
  {
    auto const data     = std::list<int>{2, 4, 6, 8, 10, 12};
    auto const expected = triangular_sum(data);
    check(data, std::plus<long>(), std::optional(static_cast<long>(expected)));
  }

  {
    auto const data     = std::deque<double>{-1.1, -2.2, -3.3, -4.4, -5.5, -6.6};
    auto plus           = [](double const x, int const y) { return x + y; };
    auto const expected = -21.1; // -5.5 + int(- 6.6) = -5.5 +  -6 = -11.5
                                 // -4.4 + int(-11.5) = -4.4 + -11 = -15.4
                                 // -3.3 + int(-15.4) = -3.3 + -15 = -18.3
                                 // -2.2 + int(-18.3) = -2.2 + -18 = -20.2
                                 // -1.1 + int(-20.2) = -1.1 + -20 = -21.1.
    check(data, plus, std::optional(expected));
  }

  {
    auto const data     = std::set<int>{2, 4, 6, 8, 10, 12};
    auto const expected = triangular_sum(data);
    check(data, std::plus<long>(), std::optional(static_cast<long>(expected)));
  }
}

int main(int, char**) {
  test_case();
  static_assert(test_case());
  runtime_only_test_case();
  return 0;
}
