//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <ranges>

#include <cassert>
#include <type_traits>
#include <vector>
#include "check_assertion.h"

constexpr int buff[] = {1, 2, 3, 4};

struct DefaultConstructibleView : std::ranges::view_base {
  constexpr DefaultConstructibleView() : begin_(buff), end_(buff + 4) {}
  constexpr int const* begin() const { return begin_; }
  constexpr int const* end() const { return end_; }

private:
  int const* begin_;
  int const* end_;
};

struct NoDefaultView : std::ranges::view_base {
  NoDefaultView() = delete;
  int* begin() const;
  int* end() const;
};

struct NoexceptView : std::ranges::view_base {
  NoexceptView() noexcept;
  int const* begin() const;
  int const* end() const;
};

struct HelperView : std::ranges::view_base {
  constexpr HelperView(const int* begin, const int* end) : begin_(begin), end_(end) {}
  constexpr int const* begin() const { return begin_; }
  constexpr int const* end() const { return end_; }

private:
  int const* begin_;
  int const* end_;
};

constexpr void test_with_one_view() {
  {
    using View = std::ranges::concat_view<DefaultConstructibleView>;
    View view;
    auto it  = view.begin();
    auto end = view.end();
    assert(*it++ == 1);
    assert(*it++ == 2);
    assert(*it++ == 3);
    assert(*it++ == 4);
    assert(it == end);
  }
}

constexpr void test_with_more_than_one_view() {
  {
    using View = std::ranges::concat_view<HelperView, HelperView>;
    int arr1[] = {1, 2};
    int arr2[] = {3, 4};
    HelperView range1(arr1, arr1 + 2);
    HelperView range2(arr2, arr2 + 2);
    View view(range1, range2);
    auto it  = view.begin();
    auto end = view.end();
    assert(*it++ == 1);
    assert(*it++ == 2);
    assert(*it++ == 3);
    assert(*it++ == 4);
    assert(it == end);
  }
}

constexpr bool tests() {
  test_with_one_view();
  test_with_more_than_one_view();

  // Check cases where the default constructor isn't provided
  { static_assert(!std::is_default_constructible_v<std::ranges::concat_view<NoDefaultView >>); }

  // Check noexcept-ness
  {
    {
      using View = std::ranges::concat_view<DefaultConstructibleView>;
      static_assert(!noexcept(View()));
    }
    {
      using View = std::ranges::concat_view<NoexceptView>;
      static_assert(noexcept(View()));
    }
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
