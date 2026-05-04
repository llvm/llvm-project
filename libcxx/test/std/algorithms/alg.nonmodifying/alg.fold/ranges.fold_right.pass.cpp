//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: std-at-least-c++23

// template<bidirectional_iterator I, sentinel_for<I> S, class T = iter_value_t<I>,
//         indirectly-binary-right-foldable<T, I> F>
//   constexpr auto ranges::fold_right(I first, S last, T init, F f);

// template<bidirectional_range R, class T = range_value_t<R>,
//         indirectly-binary-right-foldable<T, iterator_t<R>> F>
//   constexpr auto ranges::fold_right(R&& r, T init, F f);

#include <algorithm>
#include <cassert>
#include <concepts>
#include <deque>
#include <functional>
#include <iterator>
#include <list>
#include <ranges>
#include <set>
#include <string_view>
#include <string>
#include <vector>

#include "test_macros.h"
#include "test_range.h"
#include "invocable_with_telemetry.h"
#include "maths.h"

using std::ranges::fold_right;

template <std::ranges::input_range R, class T, class F, std::equality_comparable Expected>
  requires std::copyable<R>
constexpr void check_iterator(R& r, T const& init, F f, Expected const& expected) {
  {
    std::same_as<Expected> decltype(auto) result = fold_right(r.begin(), r.end(), init, f);
    assert(result == expected);
  }

  {
    auto telemetry                               = invocable_telemetry();
    auto f2                                      = invocable_with_telemetry(f, telemetry);
    std::same_as<Expected> decltype(auto) result = fold_right(r.begin(), r.end(), init, f2);
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
    std::same_as<Expected> decltype(auto) result = fold_right(r, init, f);
    assert(result == expected);
  }

  {
    auto telemetry                               = invocable_telemetry();
    auto f2                                      = invocable_with_telemetry(f, telemetry);
    std::same_as<Expected> decltype(auto) result = fold_right(r, init, f2);
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
    auto r2                                      = r;
    std::same_as<Expected> decltype(auto) result = fold_right(std::move(r2), init, f);
    assert(result == expected);
  }

  {
    auto telemetry                               = invocable_telemetry();
    auto f2                                      = invocable_with_telemetry(f, telemetry);
    auto r2                                      = r;
    std::same_as<Expected> decltype(auto) result = fold_right(std::move(r2), init, f2);
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
}

constexpr void common_range_test_case() {
  auto const data = std::vector<int>{1, 2, 3, 4};
  check(data, 0, std::plus(), triangular_sum(data));
  check(data, 1, std::multiplies(), factorial(data.back()));

  auto multiply_with_next = [n = 1](auto const x, auto const y) mutable {
    auto const result = x * y * n;
    n                 = x;
    return static_cast<std::size_t>(result);
  };
  check(data, 1, multiply_with_next, factorial(data.size()) * factorial(data.size()));

  auto fib = [n = 1](auto x, auto) mutable {
    auto old_x = x;
    x += n;
    n = old_x;
    return x;
  };
  check(data, 0, fib, fibonacci(data.back()));
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
    check(data, 0, std::plus<long>(), static_cast<long>(expected));
  }

  {
    auto const data     = std::deque<double>{-1.1, -2.2, -3.3, -4.4, -5.5, -6.6};
    auto plus           = [](double const x, int const y) { return x + y; };
    auto const expected = -21.1; // -6.6 + int(  0.0) = -6.6 +   0 =  -6.6
                                 // -5.5 + int(- 6.6) = -5.5 +  -6 = -11.5
                                 // -4.4 + int(-11.5) = -4.4 + -11 = -15.4
                                 // -3.3 + int(-15.4) = -3.3 + -15 = -18.3
                                 // -2.2 + int(-18.3) = -2.2 + -18 = -20.2
                                 // -1.1 + int(-20.2) = -1.1 + -20 = -21.1.
    check(data, 0.0, plus, expected);
  }

  {
    auto const data     = std::set<int>{2, 4, 6, 8, 10, 12};
    auto const expected = triangular_sum(data);
    check(data, 0, std::plus<long>(), static_cast<long>(expected));
  }
}

int main(int, char**) {
  test_case();
  static_assert(test_case());
  runtime_only_test_case();
  return 0;
}
