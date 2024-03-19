//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// std::views::chunk

#include <algorithm>
#include <cassert>
#include <concepts>
#include <ranges>
#include <type_traits>
#include <iostream>

#include "test_iterators.h"
#include "test_range.h"

struct Range : std::ranges::view_base {
  using Iterator = forward_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) {}
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Sentinel end() const { return Sentinel(Iterator(end_)); }

private:
  int* begin_;
  int* end_;
};

template <typename View>
constexpr void compare_views(View v, std::initializer_list<std::initializer_list<int>> list) {
  auto b1 = v.begin();
  auto e1 = v.end();
  auto b2 = list.begin();
  auto e2 = list.end();
  for (; b1 != e1 && b2 != e2; ++b1, ++b2) {
    const bool eq = std::ranges::equal(*b1, *b2, [](const auto a, const auto b) {
      assert(a == b);
      return true;
    });
    assert(eq);
  }
  assert(b1 == e1);
  assert(b2 == e2);
}

template <class T>
constexpr const T&& as_const_rvalue(T&& t) {
  return static_cast<T const&&>(t);
}

constexpr bool test() {
  constexpr int N = 10;
  int buf[N]      = {1, 1, 4, 5, 1, 4, 1, 9, 1, 9};

  // Test range adaptor object
  {
    using RangeAdaptorObject = decltype(std::views::chunk);
    static_assert(std::is_const_v<RangeAdaptorObject>);

    // The type of a customization point object, ignoring cv-qualifiers, shall model semiregular
    static_assert(std::semiregular<std::remove_const_t<RangeAdaptorObject>>);
  }

  // Test `views::chunk(n)(range)`
  {
    using Result = std::ranges::chunk_view<Range>;
    const Range range(buf, buf + N);

    {
      // `views::chunk(n)` - &&
      std::same_as<Result> decltype(auto) result = std::views::chunk(3)(range);
      compare_views(result, {{1, 1, 4}, {5, 1, 4}, {1, 9, 1}, {9}});
    }
    {
      // `views::chunk(n)` - const&&
      std::same_as<Result> decltype(auto) result = as_const_rvalue(std::views::chunk(3))(range);
      compare_views(result, {{1, 1, 4}, {5, 1, 4}, {1, 9, 1}, {9}});
    }
    {
      // `views::chunk(n)` - &
      auto partial                               = std::views::chunk(3);
      std::same_as<Result> decltype(auto) result = partial(range);
      compare_views(result, {{1, 1, 4}, {5, 1, 4}, {1, 9, 1}, {9}});
    }
    {
      // `views::chunk(n)` - const&
      const auto partial                         = std::views::chunk(3);
      std::same_as<Result> decltype(auto) result = partial(range);
      compare_views(result, {{1, 1, 4}, {5, 1, 4}, {1, 9, 1}, {9}});
    }
  }

  // Test `range | views::chunk(n)`
  {
    using Result = std::ranges::chunk_view<Range>;
    const Range range(buf, buf + N);

    {
      // `views::chunk(n)` - &&
      std::same_as<Result> decltype(auto) result = range | std::views::chunk(3);
      compare_views(result, {{1, 1, 4}, {5, 1, 4}, {1, 9, 1}, {9}});
    }
    {
      // `views::chunk(n)` - const&&
      std::same_as<Result> decltype(auto) result = range | as_const_rvalue(std::views::chunk(3));
      compare_views(result, {{1, 1, 4}, {5, 1, 4}, {1, 9, 1}, {9}});
    }
    {
      // `views::chunk(n)` - &
      auto partial                               = std::views::chunk(3);
      std::same_as<Result> decltype(auto) result = range | partial;
      compare_views(result, {{1, 1, 4}, {5, 1, 4}, {1, 9, 1}, {9}});
    }
    {
      // `views::chunk(n)` - const&
      const auto partial                         = std::views::chunk(3);
      std::same_as<Result> decltype(auto) result = range | partial;
      compare_views(result, {{1, 1, 4}, {5, 1, 4}, {1, 9, 1}, {9}});
    }
  }

  // Test `views::chunk(range, n)` range adaptor object
  {
    using Result = std::ranges::chunk_view<Range>;
    const Range range(buf, buf + N);

    {
      // `views::chunk` - &&
      auto range_adaptor                         = std::views::chunk;
      std::same_as<Result> decltype(auto) result = std::move(range_adaptor)(range, 3);
      compare_views(result, {{1, 1, 4}, {5, 1, 4}, {1, 9, 1}, {9}});
    }
    {
      // `views::chunk` - const&&
      auto const range_adaptor                   = std::views::chunk;
      std::same_as<Result> decltype(auto) result = std::move(range_adaptor)(range, 3);
      compare_views(result, {{1, 1, 4}, {5, 1, 4}, {1, 9, 1}, {9}});
    }
    {
      // `views::chunk` - &
      auto range_adaptor                         = std::views::chunk;
      std::same_as<Result> decltype(auto) result = range_adaptor(range, 3);
      compare_views(result, {{1, 1, 4}, {5, 1, 4}, {1, 9, 1}, {9}});
    }
    {
      // `views::chunk` - const&
      auto const range_adaptor                   = std::views::chunk;
      std::same_as<Result> decltype(auto) result = range_adaptor(range, 3);
      compare_views(result, {{1, 1, 4}, {5, 1, 4}, {1, 9, 1}, {9}});
    }
  }

  // Test that it's possible to call `std::views::chunk` with any single argument as long as the resulting closure is
  // never invoked. There is no good use case for it, but it's valid.
  {
    struct X {};
    [[maybe_unused]] auto partial = std::views::chunk(X{});
  }

  // Test SFINAE friendliness
  {
    struct NotAView {};

    static_assert(!CanBePiped<Range, decltype(std::views::chunk)>);
    static_assert(CanBePiped<Range, decltype(std::views::chunk(N))>);
    static_assert(!CanBePiped<NotAView, decltype(std::views::chunk(N))>);
    static_assert(!CanBePiped<std::initializer_list<int>, decltype(std::views::chunk(N))>);

    static_assert(!std::is_invocable_v<decltype(std::views::chunk)>);
    static_assert(std::is_invocable_v<decltype(std::views::chunk), Range, std::ranges::range_difference_t<Range>>);
    static_assert(!std::is_invocable_v<decltype(std::views::chunk), NotAView, std::ranges::range_difference_t<Range>>);
  }

  { static_assert(std::is_same_v<decltype(std::ranges::views::chunk), decltype(std::views::chunk)>); }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
