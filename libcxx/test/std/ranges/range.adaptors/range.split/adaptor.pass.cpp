//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::split

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <string_view>
#include <utility>

#include "test_iterators.h"

template <class View, class T>
concept CanBePiped = requires (View&& view, T&& t) {
  { std::forward<View>(view) | std::forward<T>(t) };
};

struct SomeView : std::ranges::view_base {
  const std::string_view* v_;
  constexpr SomeView(const std::string_view& v) : v_(&v) {}
  constexpr auto begin() const { return v_->begin(); }
  constexpr auto end() const { return v_->end(); }
};

struct NotAView { };

static_assert(!std::is_invocable_v<decltype(std::views::split)>);
static_assert(!std::is_invocable_v<decltype(std::views::split), SomeView, NotAView>);
static_assert(!std::is_invocable_v<decltype(std::views::split), NotAView, SomeView>);
static_assert( std::is_invocable_v<decltype(std::views::split), SomeView, SomeView>);

// Regression test for #75002, views::split shouldn't be a range adaptor closure
static_assert(!CanBePiped<SomeView&, decltype(std::views::split)>);
static_assert(!CanBePiped<char (&)[10], decltype(std::views::split)>);
static_assert(!CanBePiped<char (&&)[10], decltype(std::views::split)>);
static_assert(!CanBePiped<NotAView, decltype(std::views::split)>);

static_assert(CanBePiped<SomeView&, decltype(std::views::split('x'))>);
static_assert(CanBePiped<char (&)[10], decltype(std::views::split('x'))>);
static_assert(!CanBePiped<char (&&)[10], decltype(std::views::split('x'))>);
static_assert(!CanBePiped<NotAView, decltype(std::views::split('x'))>);

static_assert(std::same_as<decltype(std::views::split), decltype(std::ranges::views::split)>);

constexpr bool test() {
  std::string_view input = "abc";
  std::string_view sep = "a";

  // Test that `std::views::split` is a range adaptor.

  // Test `views::split(input, sep)`.
  {
    SomeView view(input);

    using Result = std::ranges::split_view<SomeView, std::string_view>;
    std::same_as<Result> decltype(auto) result = std::views::split(view, sep);
    assert(result.base().begin() == input.begin());
    assert(result.base().end() == input.end());
  }

  // Test `views::split(sep)(input)`.
  {
    SomeView view(input);

    using Result = std::ranges::split_view<SomeView, std::string_view>;
    std::same_as<Result> decltype(auto) result = std::views::split(sep)(view);
    assert(result.base().begin() == input.begin());
    assert(result.base().end() == input.end());
  }

  // Test `view | views::split`.
  {
    SomeView view(input);

    using Result = std::ranges::split_view<SomeView, std::string_view>;
    std::same_as<Result> decltype(auto) result = view | std::views::split(sep);
    assert(result.base().begin() == input.begin());
    assert(result.base().end() == input.end());
  }

  // Test `adaptor | views::split`.
  {
    SomeView view(input);
    auto f = [](char c) { return c; };
    auto partial = std::views::transform(f) | std::views::split(sep);

    using Result = std::ranges::split_view<std::ranges::transform_view<SomeView, decltype(f)>, std::string_view>;
    std::same_as<Result> decltype(auto) result = partial(view);
    assert(result.base().base().begin() == input.begin());
    assert(result.base().base().end() == input.end());
  }

  // Test `views::split | adaptor`.
  {
    SomeView view(input);
    auto f = [](auto v) { return v; };
    auto partial = std::views::split(sep) | std::views::transform(f);

    using Result = std::ranges::transform_view<std::ranges::split_view<SomeView, std::string_view>, decltype(f)>;
    std::same_as<Result> decltype(auto) result = partial(view);
    assert(result.base().base().begin() == input.begin());
    assert(result.base().base().end() == input.end());
  }

  // Test that one can call `std::views::split` with arbitrary stuff, as long as we
  // don't try to actually complete the call by passing it a range.
  //
  // That makes no sense and we can't do anything with the result, but it's valid.
  {
    struct X { };
    [[maybe_unused]] auto partial = std::views::split(X{});
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
