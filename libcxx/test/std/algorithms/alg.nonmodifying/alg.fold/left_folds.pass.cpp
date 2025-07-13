//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// MSVC warning C4244: 'argument': conversion from 'double' to 'const int', possible loss of data
// ADDITIONAL_COMPILE_FLAGS(cl-style-warnings): /wd4244

// template<input_iterator I, sentinel_for<I> S, class T,
//          indirectly-binary-left-foldable<T, I> F>
//   constexpr see below ranges::fold_left_with_iter(I first, S last, T init, F f);
//
// template<input_range R, class T, indirectly-binary-left-foldable<T, iterator_t<R>> F>
//   constexpr see below ranges::fold_left_with_iter(R&& r, T init, F f);

// template<input_iterator I, sentinel_for<I> S, class T,
//          indirectly-binary-left-foldable<T, I> F>
//   constexpr see below ranges::fold_left(I first, S last, T init, F f);
//
// template<input_range R, class T, indirectly-binary-left-foldable<T, iterator_t<R>> F>
//   constexpr see below ranges::fold_left(R&& r, T init, F f);

#include <algorithm>
#include <cassert>
#include <concepts>
#include <deque>
#include <forward_list>
#include <functional>
#include <iterator>
#include <list>
#include <ranges>
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

using std::ranges::fold_left;
using std::ranges::fold_left_with_iter;

template <class Result, class Range, class T>
concept is_in_value_result =
    std::same_as<Result, std::ranges::fold_left_with_iter_result<std::ranges::iterator_t<Range>, T>>;

template <class Result, class T>
concept is_dangling_with = std::same_as<Result, std::ranges::fold_left_with_iter_result<std::ranges::dangling, T>>;

struct Integer {
  int value;

  constexpr Integer(int const x) : value(x) {}

  constexpr Integer plus(int const x) const { return Integer{value + x}; }

  friend constexpr bool operator==(Integer const& x, Integer const& y) = default;
};

template <std::ranges::input_range R, class T, class F, std::equality_comparable Expected>
  requires std::copyable<R>
constexpr void check_iterator(R& r, T const& init, F f, Expected const& expected) {
  {
    is_in_value_result<R, Expected> decltype(auto) result = fold_left_with_iter(r.begin(), r.end(), init, f);
    assert(result.in == r.end());
    assert(result.value == expected);
  }

  {
    auto telemetry                                        = invocable_telemetry();
    auto f2                                               = invocable_with_telemetry(f, telemetry);
    is_in_value_result<R, Expected> decltype(auto) result = fold_left_with_iter(r.begin(), r.end(), init, f2);
    assert(result.in == r.end());
    assert(result.value == expected);
    assert(telemetry.invocations == std::ranges::distance(r));
    assert(telemetry.moves == 0);
    assert(telemetry.copies == 1);
  }

  {
    std::same_as<Expected> decltype(auto) result = fold_left(r.begin(), r.end(), init, f);
    assert(result == expected);
  }

  {
    auto telemetry                               = invocable_telemetry();
    auto f2                                      = invocable_with_telemetry(f, telemetry);
    std::same_as<Expected> decltype(auto) result = fold_left(r.begin(), r.end(), init, f2);
    assert(result == expected);
    assert(telemetry.invocations == std::ranges::distance(r));
    assert(telemetry.moves == 0);
    assert(telemetry.copies == 1);
  }
}

template <std::ranges::input_range R, class T, class F, std::equality_comparable Expected>
  requires std::copyable<R>
constexpr void check_lvalue_range(R& r, T const& init, F f, Expected const& expected) {
  {
    is_in_value_result<R, Expected> decltype(auto) result = fold_left_with_iter(r, init, f);
    assert(result.in == r.end());
    assert(result.value == expected);
  }

  {
    auto telemetry                               = invocable_telemetry();
    auto f2                                      = invocable_with_telemetry(f, telemetry);
    std::same_as<Expected> decltype(auto) result = fold_left(r, init, f2);
    assert(result == expected);
    assert(telemetry.invocations == std::ranges::distance(r));
    assert(telemetry.moves == 0);
    assert(telemetry.copies == 1);
  }

  {
    std::same_as<Expected> decltype(auto) result = fold_left(r, init, f);
    assert(result == expected);
  }

  {
    auto telemetry                               = invocable_telemetry();
    auto f2                                      = invocable_with_telemetry(f, telemetry);
    std::same_as<Expected> decltype(auto) result = fold_left(r, init, f2);
    assert(result == expected);
    assert(telemetry.invocations == std::ranges::distance(r));
    assert(telemetry.moves == 0);
    assert(telemetry.copies == 1);
  }
}

template <std::ranges::input_range R, class T, class F, std::equality_comparable Expected>
  requires std::copyable<R>
constexpr void check_rvalue_range(R& r, T const& init, F f, Expected const& expected) {
  {
    auto r2                                          = r;
    is_dangling_with<Expected> decltype(auto) result = fold_left_with_iter(std::move(r2), init, f);
    assert(result.value == expected);
  }

  {
    auto telemetry                                   = invocable_telemetry();
    auto f2                                          = invocable_with_telemetry(f, telemetry);
    auto r2                                          = r;
    is_dangling_with<Expected> decltype(auto) result = fold_left_with_iter(std::move(r2), init, f2);
    assert(result.value == expected);
    assert(telemetry.invocations == std::ranges::distance(r));
    assert(telemetry.moves == 0);
    assert(telemetry.copies == 1);
  }

  {
    auto r2                                      = r;
    std::same_as<Expected> decltype(auto) result = fold_left(std::move(r2), init, f);
    assert(result == expected);
  }

  {
    auto telemetry                               = invocable_telemetry();
    auto f2                                      = invocable_with_telemetry(f, telemetry);
    auto r2                                      = r;
    std::same_as<Expected> decltype(auto) result = fold_left(std::move(r2), init, f2);
    assert(result == expected);
    assert(telemetry.invocations == std::ranges::distance(r));
    assert(telemetry.moves == 0);
    assert(telemetry.copies == 1);
  }
}

template <std::ranges::input_range R, class T, class F, std::equality_comparable Expected>
  requires std::copyable<R>
constexpr void check(R r, T const& init, F f, Expected const& expected) {
  check_iterator(r, init, f, expected);
  check_lvalue_range(r, init, f, expected);
  check_rvalue_range(r, init, f, expected);
}

constexpr void empty_range_test_case() {
  auto const data = std::vector<int>{};
  check(data, 100, std::plus(), 100);
  check(data, -100, std::multiplies(), -100);

  check(data | std::views::take_while([](auto) { return false; }), 1.23, std::plus(), 1.23);
  check(data, Integer(52), &Integer::plus, Integer(52));
}

constexpr void common_range_test_case() {
  auto const data = std::vector<int>{1, 2, 3, 4};
  check(data, 0, std::plus(), triangular_sum(data));
  check(data, 1, std::multiplies(), factorial(data.back()));

  auto multiply_with_prev = [n = 1](auto const x, auto const y) mutable {
    auto const result = x * y * n;
    n                 = y;
    return static_cast<std::size_t>(result);
  };
  check(data, 1, multiply_with_prev, factorial(data.size()) * factorial(data.size() - 1));

  auto fib = [n = 1](auto x, auto) mutable {
    auto old_x = x;
    x += n;
    n = old_x;
    return x;
  };
  check(data, 0, fib, fibonacci(data.back()));

  check(data, Integer(0), &Integer::plus, Integer(triangular_sum(data)));
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
    check(range, 0, std::plus(), triangular_sum(range));
  }

  {
    auto data           = std::string("five three two six one four");
    auto to_string_view = [](auto&& r) {
      auto const n = std::ranges::distance(r);
      return std::string_view(&*r.begin(), n);
    };
    auto range =
        std::views::lazy_split(data, ' ') | std::views::transform(to_string_view) | std::views::transform(parse);
    check(range, 0, std::plus(), triangular_sum(range));
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
    constexpr auto expected = "WindmillShellsOrangeSyrupBaratieCocoyashiLoguetown";
    auto const init         = std::string("Windmill");

    {
      auto input = std::istringstream(raw_data);
      auto data  = std::views::istream<std::string>(input);
      is_in_value_result<std::ranges::basic_istream_view<std::string, char>, std::string> decltype(auto) result =
          fold_left_with_iter(data.begin(), data.end(), init, std::plus());

      assert(result.in == data.end());
      assert(result.value == expected);
    }

    {
      auto input = std::istringstream(raw_data);
      auto data  = std::views::istream<std::string>(input);
      is_in_value_result<std::ranges::basic_istream_view<std::string, char>, std::string> decltype(auto) result =
          fold_left_with_iter(data, init, std::plus());
      assert(result.in == data.end());
      assert(result.value == expected);
    }

    {
      auto input = std::istringstream(raw_data);
      auto data  = std::views::istream<std::string>(input);
      assert(fold_left(data.begin(), data.end(), init, std::plus()) == expected);
    }

    {
      auto input = std::istringstream(raw_data);
      auto data  = std::views::istream<std::string>(input);
      assert(fold_left(data, init, std::plus()) == expected);
    }
  }
#endif
  {
    auto const data     = std::forward_list<int>{1, 3, 5, 7, 9};
    auto const n        = std::ranges::distance(data);
    auto const expected = static_cast<float>(n * n); // sum of n consecutive odd numbers = n^2
    check(data, 0.0f, std::plus(), expected);
  }

  {
    auto const data     = std::list<int>{2, 4, 6, 8, 10, 12};
    auto const expected = triangular_sum(data);
    check(data, 0, std::plus<long>(), static_cast<long>(expected));
  }

  {
    auto const data     = std::deque<double>{-1.1, -2.2, -3.3, -4.4, -5.5, -6.6};
    auto plus           = [](int const x, double const y) { return x + y; };
    auto const expected = -21.6; // int(  0.0) + -1.1 =   0 + -1.1 =  -1.1
                                 // int(- 1.1) + -2.2 = - 1 + -2.2 =  -3.2
                                 // int(- 3.2) + -3.3 = - 3 + -3.3 =  -6.3
                                 // int(- 6.3) + -4.4 = - 6 + -4.4 = -10.4
                                 // int(-10.4) + -5.5 = -10 + -5.5 = -15.5
                                 // int(-15.5) + -6.6 = -15 + -6.6 = -21.6.
    check(data, 0.0, plus, expected);
  }
}

int main(int, char**) {
  test_case();
  static_assert(test_case());
  runtime_only_test_case();
  return 0;
}
