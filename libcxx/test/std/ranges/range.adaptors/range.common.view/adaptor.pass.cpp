//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::common

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <utility>

#include "test_iterators.h"
#include "types.h"

template <class View, class T>
concept CanBePiped = requires (View&& view, T&& t) {
  { std::forward<View>(view) | std::forward<T>(t) };
};

constexpr bool test() {
  int buf[] = {1, 2, 3};

  // views::common(r) is equivalent to views::all(r) if r is a common_range
  {
    {
      CommonView view(buf, buf + 3);
      std::same_as<CommonView> auto result = std::views::common(view);
      assert(result.begin_ == buf);
      assert(result.end_ == buf + 3);
    }
    {
      using NotAView = std::array<int, 3>;
      NotAView arr = {1, 2, 3};
      std::same_as<std::ranges::ref_view<NotAView>> auto result = std::views::common(arr);
      assert(result.begin() == arr.begin());
      assert(result.end() == arr.end());
    }
  }

  // Otherwise, views::common(r) is equivalent to ranges::common_view{r}
  {
    NonCommonView view(buf, buf + 3);
    std::same_as<std::ranges::common_view<NonCommonView>> auto result = std::views::common(view);
    assert(result.base().begin_ == buf);
    assert(result.base().end_ == buf + 3);
  }

  // Test that std::views::common is a range adaptor
  {
    using SomeView = NonCommonView;

    // Test `v | views::common`
    {
      SomeView view(buf, buf + 3);
      std::same_as<std::ranges::common_view<SomeView>> auto result = view | std::views::common;
      assert(result.base().begin_ == buf);
      assert(result.base().end_ == buf + 3);
    }

    // Test `adaptor | views::common`
    {
      SomeView view(buf, buf + 3);
      auto f = [](int i) { return i; };
      auto const partial = std::views::transform(f) | std::views::common;
      using Result = std::ranges::common_view<std::ranges::transform_view<SomeView, decltype(f)>>;
      std::same_as<Result> auto result = partial(view);
      assert(result.base().base().begin_ == buf);
      assert(result.base().base().end_ == buf + 3);
    }

    // Test `views::common | adaptor`
    {
      SomeView view(buf, buf + 3);
      auto f = [](int i) { return i; };
      auto const partial = std::views::common | std::views::transform(f);
      using Result = std::ranges::transform_view<std::ranges::common_view<SomeView>, decltype(f)>;
      std::same_as<Result> auto result = partial(view);
      assert(result.base().base().begin_ == buf);
      assert(result.base().base().end_ == buf + 3);
    }

    // Check SFINAE friendliness
    {
      struct NotAView { };
      static_assert(!std::is_invocable_v<decltype(std::views::common)>);
      static_assert(!std::is_invocable_v<decltype(std::views::common), NotAView>);
      static_assert( CanBePiped<SomeView&,   decltype(std::views::common)>);
      static_assert( CanBePiped<int(&)[10],  decltype(std::views::common)>);
      static_assert(!CanBePiped<int(&&)[10], decltype(std::views::common)>);
      static_assert(!CanBePiped<NotAView,    decltype(std::views::common)>);
    }
  }

  {
    static_assert(std::same_as<decltype(std::views::common), decltype(std::ranges::views::common)>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
