//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: std-at-least-c++23

// template<input_iterator I, sentinel_for<I> S,
//          indirectly-binary-left-foldable<iter_value_t<I>, I> F>
//   requires constructible_from<iter_value_t<I>, iter_reference_t<I>>
//   constexpr see below ranges::fold_left_first_with_iter(I first, S last, F f);

// template<input_range R, indirectly-binary-left-foldable<range_value_t<R>, iterator_t<R>> F>
//   requires constructible_from<range_value_t<R>, range_reference_t<R>>
//   constexpr see below ranges::fold_left_first_with_iter(R&& r, F f);

// template<input_iterator I, sentinel_for<I> S,
//          indirectly-binary-left-foldable<iter_value_t<I>, I> F>
//   requires constructible_from<iter_value_t<I>, iter_reference_t<I>>
//   constexpr auto ranges::fold_left_first(I first, S last, F f);

// template<input_range R, indirectly-binary-left-foldable<range_value_t<R>, iterator_t<R>> F>
//   requires constructible_from<range_value_t<R>, range_reference_t<R>>
//   constexpr auto ranges::fold_left_first(R&& r, F f);

#include <algorithm>
#include <cassert>
#include <concepts>
#include <deque>
#include <forward_list>
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

#if !defined(TEST_HAS_NO_LOCALIZATION)
#  include <sstream>
#endif

using std::ranges::fold_left_first;
using std::ranges::fold_left_first_with_iter;

template <class Result, class Range, class T>
concept is_in_value_result =
    std::same_as<Result, std::ranges::fold_left_first_with_iter_result<std::ranges::iterator_t<Range>, T>>;

template <class Result, class T>
concept is_dangling_with =
    std::same_as<Result, std::ranges::fold_left_first_with_iter_result<std::ranges::dangling, T>>;

template <std::ranges::input_range R, class F, std::equality_comparable Expected>
  requires std::copyable<R>
constexpr void check_iterator(R& r, F f, std::optional<Expected> const& expected) {
  {
    is_in_value_result<R, std::optional<Expected>> decltype(auto) result =
        fold_left_first_with_iter(r.begin(), r.end(), f);
    assert(result.in == r.end());
    assert(result.value == expected);
  }

  {
    auto telemetry = invocable_telemetry();
    auto f2        = invocable_with_telemetry(f, telemetry);
    is_in_value_result<R, std::optional<Expected>> decltype(auto) result =
        fold_left_first_with_iter(r.begin(), r.end(), f2);
    assert(result.in == r.end());
    assert(result.value == expected);
    if (expected.has_value()) {
      assert(telemetry.invocations == std::ranges::distance(r) - 1);
      assert(telemetry.moves == 0);
      assert(telemetry.copies == 1);
    }
  }

  {
    std::same_as<std::optional<Expected>> decltype(auto) result = fold_left_first(r.begin(), r.end(), f);
    assert(result == expected);
  }

  {
    auto telemetry                                              = invocable_telemetry();
    auto f2                                                     = invocable_with_telemetry(f, telemetry);
    std::same_as<std::optional<Expected>> decltype(auto) result = fold_left_first(r.begin(), r.end(), f2);
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
constexpr void check_lvalue_range(R& r, F f, std::optional<Expected> const& expected) {
  {
    is_in_value_result<R, std::optional<Expected>> decltype(auto) result = fold_left_first_with_iter(r, f);
    assert(result.in == r.end());
    assert(result.value == expected);
  }

  {
    auto telemetry                                              = invocable_telemetry();
    auto f2                                                     = invocable_with_telemetry(f, telemetry);
    std::same_as<std::optional<Expected>> decltype(auto) result = fold_left_first(r, f2);
    assert(result == expected);
    if (expected.has_value()) {
      assert(telemetry.invocations == std::ranges::distance(r) - 1);
      assert(telemetry.moves == 0);
      assert(telemetry.copies == 1);
    }
  }

  {
    std::same_as<std::optional<Expected>> decltype(auto) result = fold_left_first(r, f);
    assert(result == expected);
  }

  {
    auto telemetry                                              = invocable_telemetry();
    auto f2                                                     = invocable_with_telemetry(f, telemetry);
    std::same_as<std::optional<Expected>> decltype(auto) result = fold_left_first(r, f2);
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
    auto r2                                                         = r;
    is_dangling_with<std::optional<Expected>> decltype(auto) result = fold_left_first_with_iter(std::move(r2), f);
    assert(result.value == expected);
  }

  {
    auto telemetry                                                  = invocable_telemetry();
    auto f2                                                         = invocable_with_telemetry(f, telemetry);
    auto r2                                                         = r;
    is_dangling_with<std::optional<Expected>> decltype(auto) result = fold_left_first_with_iter(std::move(r2), f2);
    assert(result.value == expected);
    if (expected.has_value()) {
      assert(telemetry.invocations == std::ranges::distance(r) - 1);
      assert(telemetry.moves == 0);
      assert(telemetry.copies == 1);
    }
  }

  {
    auto r2                                                     = r;
    std::same_as<std::optional<Expected>> decltype(auto) result = fold_left_first(std::move(r2), f);
    assert(result == expected);
  }

  {
    auto telemetry                                              = invocable_telemetry();
    auto f2                                                     = invocable_with_telemetry(f, telemetry);
    auto r2                                                     = r;
    std::same_as<std::optional<Expected>> decltype(auto) result = fold_left_first(std::move(r2), f2);
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

  auto fib = [n = 0](auto x, auto) mutable {
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

// Most containers aren't constexpr
void runtime_only_test_case() {
#if !defined(TEST_HAS_NO_LOCALIZATION)
  { // istream_view is a genuine input range and needs specific handling.
    constexpr auto raw_data = "Shells Orange Syrup Baratie Cocoyashi Loguetown";
    constexpr auto expected = "ShellsOrangeSyrupBaratieCocoyashiLoguetown";

    {
      auto input = std::istringstream(raw_data);
      auto data  = std::views::istream<std::string>(input);
      is_in_value_result<std::ranges::basic_istream_view<std::string, char>, std::optional<std::string>> decltype(auto)
          result = fold_left_first_with_iter(data.begin(), data.end(), std::plus());

      assert(result.in == data.end());
      assert(result.value == expected);
    }

    {
      auto input = std::istringstream(raw_data);
      auto data  = std::views::istream<std::string>(input);
      is_in_value_result<std::ranges::basic_istream_view<std::string, char>, std::optional<std::string>> decltype(auto)
          result = fold_left_first_with_iter(data, std::plus());
      assert(result.in == data.end());
      assert(result.value == expected);
    }

    {
      auto input = std::istringstream(raw_data);
      auto data  = std::views::istream<std::string>(input);
      assert(fold_left_first(data.begin(), data.end(), std::plus()) == expected);
    }

    {
      auto input = std::istringstream(raw_data);
      auto data  = std::views::istream<std::string>(input);
      assert(fold_left_first(data, std::plus()) == expected);
    }
  }
#endif
  {
    auto const data     = std::forward_list<int>{1, 3, 5, 7, 9};
    auto const n        = std::ranges::distance(data);
    auto const expected = n * n; // sum of n consecutive odd numbers = n^2
    check(data, std::plus(), std::optional(static_cast<int>(expected)));
  }

  {
    auto const data     = std::list<int>{2, 4, 6, 8, 10, 12};
    auto const expected = triangular_sum(data);
    check(data, std::plus<long>(), std::optional(static_cast<long>(expected)));
  }

  {
    auto const data     = std::deque<double>{-1.1, -2.2, -3.3, -4.4, -5.5, -6.6};
    auto plus           = [](double const x, double const y) { return static_cast<int>(x) + y; };
    auto const expected = -21.6; // int(- 1.1) + -2.2 = - 1 + -2.2 =  -3.2
                                 // int(- 3.2) + -3.3 = - 3 + -3.3 =  -6.3
                                 // int(- 6.3) + -4.4 = - 6 + -4.4 = -10.4
                                 // int(-10.4) + -5.5 = -10 + -5.5 = -15.5
                                 // int(-15.5) + -6.6 = -15 + -6.6 = -21.6.
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
