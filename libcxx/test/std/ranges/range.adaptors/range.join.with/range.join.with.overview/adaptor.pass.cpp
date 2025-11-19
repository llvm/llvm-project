//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// std::views::join_with_view

#include <ranges>

#include <memory>
#include <span>
#include <string_view>
#include <utility>

#include "test_iterators.h"

template <class View, class T>
concept CanBePiped = requires(View&& view, T&& t) {
  { std::forward<View>(view) | std::forward<T>(t) };
};

struct Range : std::ranges::view_base {
  using Iterator = forward_iterator<std::string_view*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  constexpr explicit Range(std::string_view* b, std::string_view* e) : begin_(b), end_(e) {}
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Sentinel end() const { return Sentinel(Iterator(end_)); }

private:
  std::string_view* begin_;
  std::string_view* end_;
};

struct Pattern : std::ranges::view_base {
  using Iterator = forward_iterator<const char*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  static constexpr std::string_view pat{", "};

  constexpr Pattern() = default;
  constexpr Iterator begin() const { return Iterator(pat.data()); }
  constexpr Sentinel end() const { return Sentinel(Iterator(pat.data() + pat.size())); }
};

struct NonCopyablePattern : Pattern {
  NonCopyablePattern(const NonCopyablePattern&) = delete;
};

template <typename View>
constexpr void compareViews(View v, std::string_view list) {
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

constexpr int absoluteValue(int x) { return x < 0 ? -x : x; }

template <class T>
constexpr const T&& asConstRvalue(T&& t) {
  return static_cast<const T&&>(t);
}

constexpr void test_adaptor_with_pattern(std::span<std::string_view> buff) {
  // Test `views::join_with(pattern)(v)`
  {
    using Result = std::ranges::join_with_view<Range, Pattern>;
    const Range range(buff.data(), buff.data() + buff.size());
    Pattern pattern;

    {
      // 'views::join_with(pattern)' - &&
      std::same_as<Result> decltype(auto) result = std::views::join_with(pattern)(range);
      compareViews(result, "abcd, ef, ghij, kl");
    }
    {
      // 'views::join_with(pattern)' - const&&
      std::same_as<Result> decltype(auto) result = asConstRvalue(std::views::join_with(pattern))(range);
      compareViews(result, "abcd, ef, ghij, kl");
    }
    {
      // 'views::join_with(pattern)' - &
      auto partial                               = std::views::join_with(pattern);
      std::same_as<Result> decltype(auto) result = partial(range);
      compareViews(result, "abcd, ef, ghij, kl");
    }
    {
      // 'views::join_with(pattern)' - const&
      auto const partial                         = std::views::join_with(pattern);
      std::same_as<Result> decltype(auto) result = partial(range);
      compareViews(result, "abcd, ef, ghij, kl");
    }
  }

  // Test `v | views::join_with(pattern)`
  {
    using Result = std::ranges::join_with_view<Range, Pattern>;
    const Range range(buff.data(), buff.data() + buff.size());
    Pattern pattern;

    {
      // 'views::join_with(pattern)' - &&
      std::same_as<Result> decltype(auto) result = range | std::views::join_with(pattern);
      compareViews(result, "abcd, ef, ghij, kl");
    }
    {
      // 'views::join_with(pattern)' - const&&
      std::same_as<Result> decltype(auto) result = range | asConstRvalue(std::views::join_with(pattern));
      compareViews(result, "abcd, ef, ghij, kl");
    }
    {
      // 'views::join_with(pattern)' - &
      auto partial                               = std::views::join_with(pattern);
      std::same_as<Result> decltype(auto) result = range | partial;
      compareViews(result, "abcd, ef, ghij, kl");
    }
    {
      // 'views::join_with(pattern)' - const&
      auto const partial                         = std::views::join_with(pattern);
      std::same_as<Result> decltype(auto) result = range | partial;
      compareViews(result, "abcd, ef, ghij, kl");
    }
  }

  // Test `views::join_with(v, pattern)` range adaptor object
  {
    using Result = std::ranges::join_with_view<Range, Pattern>;
    const Range range(buff.data(), buff.data() + buff.size());
    Pattern pattern;

    {
      // 'views::join_with' - &&
      auto range_adaptor                         = std::views::join_with;
      std::same_as<Result> decltype(auto) result = std::move(range_adaptor)(range, pattern);
      compareViews(result, "abcd, ef, ghij, kl");
    }
    {
      // 'views::join_with' - const&&
      const auto range_adaptor                   = std::views::join_with;
      std::same_as<Result> decltype(auto) result = std::move(range_adaptor)(range, pattern);
      compareViews(result, "abcd, ef, ghij, kl");
    }
    {
      // 'views::join_with' - &
      auto range_adaptor                         = std::views::join_with;
      std::same_as<Result> decltype(auto) result = range_adaptor(range, pattern);
      compareViews(result, "abcd, ef, ghij, kl");
    }
    {
      // 'views::join_with' - const&
      const auto range_adaptor                   = std::views::join_with;
      std::same_as<Result> decltype(auto) result = range_adaptor(range, pattern);
      compareViews(result, "abcd, ef, ghij, kl");
    }
  }

  // Test `adaptor | views::join_with(pattern)`
  {
    auto pred    = [](std::string_view s) { return s.size() >= 3; };
    using Result = std::ranges::join_with_view<std::ranges::filter_view<Range, decltype(pred)>, Pattern>;
    const Range range(buff.data(), buff.data() + buff.size());
    Pattern pattern;

    {
      std::same_as<Result> decltype(auto) result = range | std::views::filter(pred) | std::views::join_with(pattern);
      compareViews(result, "abcd, ghij");
    }
    {
      const auto partial                         = std::views::filter(pred) | std::views::join_with(pattern);
      std::same_as<Result> decltype(auto) result = range | partial;
      compareViews(result, "abcd, ghij");
    }
  }
}

constexpr void test_adaptor_with_single_element(std::span<std::string_view> buff) {
  // Test `views::join_with(element)(v)`
  {
    using Result = std::ranges::join_with_view<Range, std::ranges::single_view<char>>;
    const Range range(buff.data(), buff.data() + buff.size());
    const char element = '.';

    {
      // 'views::join_with(element)' - &&
      std::same_as<Result> decltype(auto) result = std::views::join_with(element)(range);
      compareViews(result, "abcd.ef.ghij.kl");
    }
    {
      // 'views::join_with(element)' - const&&
      std::same_as<Result> decltype(auto) result = asConstRvalue(std::views::join_with(element))(range);
      compareViews(result, "abcd.ef.ghij.kl");
    }
    {
      // 'views::join_with(element)' - &
      auto partial                               = std::views::join_with(element);
      std::same_as<Result> decltype(auto) result = partial(range);
      compareViews(result, "abcd.ef.ghij.kl");
    }
    {
      // 'views::join_with(element)' - const&
      const auto partial                         = std::views::join_with(element);
      std::same_as<Result> decltype(auto) result = partial(range);
      compareViews(result, "abcd.ef.ghij.kl");
    }
  }

  // Test `v | views::join_with(element)`
  {
    using Result = std::ranges::join_with_view<Range, std::ranges::single_view<char>>;
    const Range range(buff.data(), buff.data() + buff.size());
    const char element = '.';

    {
      // 'views::join_with(element)' - &&
      std::same_as<Result> decltype(auto) result = range | std::views::join_with(element);
      compareViews(result, "abcd.ef.ghij.kl");
    }
    {
      // 'views::join_with(element)' - const&&
      std::same_as<Result> decltype(auto) result = range | asConstRvalue(std::views::join_with(element));
      compareViews(result, "abcd.ef.ghij.kl");
    }
    {
      // 'views::join_with(element)' - &
      auto partial                               = std::views::join_with(element);
      std::same_as<Result> decltype(auto) result = range | partial;
      compareViews(result, "abcd.ef.ghij.kl");
    }
    {
      // 'views::join_with(element)' - const&
      const auto partial                         = std::views::join_with(element);
      std::same_as<Result> decltype(auto) result = range | partial;
      compareViews(result, "abcd.ef.ghij.kl");
    }
  }

  // Test `views::join_with(v, element)` range adaptor object
  {
    using Result = std::ranges::join_with_view<Range, std::ranges::single_view<char>>;
    const Range range(buff.data(), buff.data() + buff.size());
    const char element = '.';

    {
      // 'views::join_with' - &&
      auto range_adaptor                         = std::views::join_with;
      std::same_as<Result> decltype(auto) result = std::move(range_adaptor)(range, element);
      compareViews(result, "abcd.ef.ghij.kl");
    }
    {
      // 'views::join_with' - const&&
      const auto range_adaptor                   = std::views::join_with;
      std::same_as<Result> decltype(auto) result = std::move(range_adaptor)(range, element);
      compareViews(result, "abcd.ef.ghij.kl");
    }
    {
      // 'views::join_with' - &
      auto range_adaptor                         = std::views::join_with;
      std::same_as<Result> decltype(auto) result = range_adaptor(range, element);
      compareViews(result, "abcd.ef.ghij.kl");
    }
    {
      // 'views::join_with' - const&
      const auto range_adaptor                   = std::views::join_with;
      std::same_as<Result> decltype(auto) result = range_adaptor(range, element);
      compareViews(result, "abcd.ef.ghij.kl");
    }
  }

  // Test `adaptor | views::join_with(element)`
  {
    auto pred = [](std::string_view s) { return s.size() >= 3; };
    using Result =
        std::ranges::join_with_view<std::ranges::filter_view<Range, decltype(pred)>, std::ranges::single_view<char>>;
    const Range range(buff.data(), buff.data() + buff.size());
    const char element = '.';

    {
      std::same_as<Result> decltype(auto) result = range | std::views::filter(pred) | std::views::join_with(element);
      compareViews(result, "abcd.ghij");
    }
    {
      const auto partial                         = std::views::filter(pred) | std::views::join_with(element);
      std::same_as<Result> decltype(auto) result = range | partial;
      compareViews(result, "abcd.ghij");
    }
  }
}

constexpr bool test() {
  std::string_view buff[] = {"abcd", "ef", "ghij", "kl"};

  // Test range adaptor object
  {
    using RangeAdaptorObject = decltype(std::views::join_with);
    static_assert(std::is_const_v<RangeAdaptorObject>);

    // The type of a customization point object, ignoring cv-qualifiers, shall model semiregular
    static_assert(std::semiregular<std::remove_const<RangeAdaptorObject>>);
  }

  test_adaptor_with_pattern(buff);
  test_adaptor_with_single_element(buff);

  // Test that one can call std::views::join_with with arbitrary stuff, as long as we
  // don't try to actually complete the call by passing it a range.
  //
  // That makes no sense and we can't do anything with the result, but it's valid.
  {
    long array[3]                 = {1, 2, 3};
    [[maybe_unused]] auto partial = std::views::join_with(std::move(array));
  }

  // Test SFINAE friendliness
  {
    struct NotAView {};

    static_assert(!CanBePiped<Range, decltype(std::views::join_with)>);
    static_assert(CanBePiped<Range, decltype(std::views::join_with(Pattern{}))>);
    static_assert(CanBePiped<Range, decltype(std::views::join_with('.'))>);
    static_assert(!CanBePiped<NotAView, decltype(std::views::join_with(Pattern{}))>);
    static_assert(!CanBePiped<NotAView, decltype(std::views::join_with('.'))>);
    static_assert(!CanBePiped<std::initializer_list<char>, decltype(std::views::join_with(Pattern{}))>);
    static_assert(!CanBePiped<std::initializer_list<char>, decltype(std::views::join_with('.'))>);
    static_assert(!CanBePiped<Range, decltype(std::views::join_with(NotAView{}))>);

    static_assert(!std::is_invocable_v<decltype(std::views::join_with)>);
    static_assert(!std::is_invocable_v<decltype(std::views::join_with), Pattern, Range>);
    static_assert(!std::is_invocable_v<decltype(std::views::join_with), char, Range>);
    static_assert(std::is_invocable_v<decltype(std::views::join_with), Range, Pattern>);
    static_assert(std::is_invocable_v<decltype(std::views::join_with), Range, char>);
    static_assert(!std::is_invocable_v<decltype(std::views::join_with), Range, Pattern, Pattern>);
    static_assert(!std::is_invocable_v<decltype(std::views::join_with), Range, char, char>);
    static_assert(!std::is_invocable_v<decltype(std::views::join_with), NonCopyablePattern>);
  }

  {
    static_assert(std::is_same_v<decltype(std::ranges::views::join_with), decltype(std::views::join_with)>);
    assert(std::addressof(std::ranges::views::join_with) == std::addressof(std::views::join_with));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
