//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// std::views::filter

#include <ranges>

#include <cassert>
#include <concepts>
#include <initializer_list>
#include <type_traits>
#include <utility>

#include "test_iterators.h"

template <class View, class T>
concept CanBePiped = requires (View&& view, T&& t) {
  { std::forward<View>(view) | std::forward<T>(t) };
};

struct NonCopyablePredicate {
  NonCopyablePredicate(NonCopyablePredicate const&) = delete;
  template <class T>
  constexpr bool operator()(T x) const { return x % 2 == 0; }
};

struct Range : std::ranges::view_base {
  using Iterator = forward_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) { }
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Sentinel end() const { return Sentinel(Iterator(end_)); }

private:
  int* begin_;
  int* end_;
};

struct Pred {
  constexpr bool operator()(int i) const { return i % 2 == 0; }
};

template <typename View>
constexpr void compareViews(View v, std::initializer_list<int> list) {
  auto b1 = v.begin();
  auto e1 = v.end();
  auto b2 = list.begin();
  auto e2 = list.end();
  for (; b1 != e1 && b2 != e2; ++b1, ++b2) {
    assert(*b1 == *b2);
  }
  assert(b1 == e1);
  assert(b2 == e2);
}

constexpr bool test() {
  int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};

  // Test `views::filter(pred)(v)`
  {
    using Result = std::ranges::filter_view<Range, Pred>;
    Range const range(buff, buff + 8);
    Pred pred;

    {
      std::same_as<Result> decltype(auto) result = std::views::filter(pred)(range);
      compareViews(result, {0, 2, 4, 6});
    }
    {
      auto const partial = std::views::filter(pred);
      std::same_as<Result> decltype(auto) result = partial(range);
      compareViews(result, {0, 2, 4, 6});
    }
  }

  // Test `v | views::filter(pred)`
  {
    using Result = std::ranges::filter_view<Range, Pred>;
    Range const range(buff, buff + 8);
    Pred pred;

    {
      std::same_as<Result> decltype(auto) result = range | std::views::filter(pred);
      compareViews(result, {0, 2, 4, 6});
    }
    {
      auto const partial = std::views::filter(pred);
      std::same_as<Result> decltype(auto) result = range | partial;
      compareViews(result, {0, 2, 4, 6});
    }
  }

  // Test `views::filter(v, pred)`
  {
    using Result = std::ranges::filter_view<Range, Pred>;
    Range const range(buff, buff + 8);
    Pred pred;

    std::same_as<Result> decltype(auto) result = std::views::filter(range, pred);
    compareViews(result, {0, 2, 4, 6});
  }

  // Test that one can call std::views::filter with arbitrary stuff, as long as we
  // don't try to actually complete the call by passing it a range.
  //
  // That makes no sense and we can't do anything with the result, but it's valid.
  {
    struct X { };
    [[maybe_unused]] auto partial = std::views::filter(X{});
  }

  // Test `adaptor | views::filter(pred)`
  {
    Range const range(buff, buff + 8);

    {
      auto pred1 = [](int i) { return i % 2 == 0; };
      auto pred2 = [](int i) { return i % 3 == 0; };
      using Result = std::ranges::filter_view<std::ranges::filter_view<Range, decltype(pred1)>, decltype(pred2)>;
      std::same_as<Result> decltype(auto) result = range | std::views::filter(pred1) | std::views::filter(pred2);
      compareViews(result, {0, 6});
    }
    {
      auto pred1 = [](int i) { return i % 2 == 0; };
      auto pred2 = [](int i) { return i % 3 == 0; };
      using Result = std::ranges::filter_view<std::ranges::filter_view<Range, decltype(pred1)>, decltype(pred2)>;
      auto const partial = std::views::filter(pred1) | std::views::filter(pred2);
      std::same_as<Result> decltype(auto) result = range | partial;
      compareViews(result, {0, 6});
    }
  }

  // Test SFINAE friendliness
  {
    struct NotAView { };
    struct NotInvocable { };

    static_assert(!CanBePiped<Range,    decltype(std::views::filter)>);
    static_assert( CanBePiped<Range,    decltype(std::views::filter(Pred{}))>);
    static_assert(!CanBePiped<NotAView, decltype(std::views::filter(Pred{}))>);
    static_assert(!CanBePiped<Range,    decltype(std::views::filter(NotInvocable{}))>);

    static_assert(!std::is_invocable_v<decltype(std::views::filter)>);
    static_assert(!std::is_invocable_v<decltype(std::views::filter), Pred, Range>);
    static_assert( std::is_invocable_v<decltype(std::views::filter), Range, Pred>);
    static_assert(!std::is_invocable_v<decltype(std::views::filter), Range, Pred, Pred>);
    static_assert(!std::is_invocable_v<decltype(std::views::filter), NonCopyablePredicate>);
  }

  {
    static_assert(std::is_same_v<decltype(std::ranges::views::filter), decltype(std::views::filter)>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
