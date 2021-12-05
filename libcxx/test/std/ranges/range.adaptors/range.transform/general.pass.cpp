//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// Some basic examples of how transform_view might be used in the wild. This is a general
// collection of sample algorithms and functions that try to mock general usage of
// this view.

#include <ranges>

#include <cctype>
#include <functional>
#include <list>
#include <numeric>
#include <string>
#include <vector>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"
#include "types.h"

template<class T, class F>
concept ValidTransformView = requires { typename std::ranges::transform_view<T, F>; };

struct BadFunction { };
static_assert( ValidTransformView<MoveOnlyView, PlusOne>);
static_assert(!ValidTransformView<Range, PlusOne>);
static_assert(!ValidTransformView<MoveOnlyView, BadFunction>);

template<std::ranges::range R>
auto toUpper(R range) {
  return std::ranges::transform_view(range, [](char c) { return std::toupper(c); });
}

template<class E1, class E2, size_t N, class Join = std::plus<E1>>
auto joinArrays(E1 (&a)[N], E2 (&b)[N], Join join = Join()) {
  return std::ranges::transform_view(a, [&a, &b, join](auto& x) {
    auto idx = (&x) - a;
    return join(x, b[idx]);
  });
}

struct NonConstView : std::ranges::view_base {
  explicit NonConstView(int *b, int *e) : b_(b), e_(e) {}
  const int *begin() { return b_; }  // deliberately non-const
  const int *end() { return e_; }    // deliberately non-const
  const int *b_;
  const int *e_;
};

int main(int, char**) {
  {
    std::vector<int> vec = {1, 2, 3, 4};
    auto transformed = std::ranges::transform_view(vec, [](int x) { return x + 42; });
    int expected[] = {43, 44, 45, 46};
    assert(std::equal(transformed.begin(), transformed.end(), expected, expected + 4));
    const auto& ct = transformed;
    assert(std::equal(ct.begin(), ct.end(), expected, expected + 4));
  }

  {
    // Test a view type that is not const-iterable.
    int a[] = {1, 2, 3, 4};
    auto transformed = NonConstView(a, a + 4) | std::views::transform([](int x) { return x + 42; });
    int expected[4] = {43, 44, 45, 46};
    assert(std::equal(transformed.begin(), transformed.end(), expected, expected + 4));
  }

  {
    int a[4] = {1, 2, 3, 4};
    int b[4] = {4, 3, 2, 1};
    auto out = joinArrays(a, b);
    int check[4] = {5, 5, 5, 5};
    assert(std::equal(out.begin(), out.end(), check, check + 4));
  }

  {
    std::string_view str = "Hello, World.";
    auto upp = toUpper(str);
    std::string_view check = "HELLO, WORLD.";
    assert(std::equal(upp.begin(), upp.end(), check.begin(), check.end()));
  }

  return 0;
}
