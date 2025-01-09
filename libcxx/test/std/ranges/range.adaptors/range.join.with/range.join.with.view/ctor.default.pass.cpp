//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// join_with_view()
//   requires default_initializable<V> && default_initializable<Pattern> = default;

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <type_traits>

static constexpr auto view = std::to_array<std::array<int, 2>>({{1, 2}, {3, 4}, {5, 6}});

struct TrivialView : std::ranges::view_base {
  int val_; // intentionally uninitialized

  constexpr auto begin() { return view.data(); }
  constexpr auto end() { return view.data() + view.size(); }
};

static_assert(std::is_trivial_v<TrivialView>);

struct NonDefaultConstructibleView : TrivialView {
  NonDefaultConstructibleView(int);
};

struct TrivialPattern : std::ranges::view_base {
  int val_; // intentionally uninitialized

  constexpr int* begin() { return &val_; }
  constexpr int* end() { return &val_ + 1; }
};

static_assert(std::is_trivial_v<TrivialPattern>);

struct NonDefaultConstructiblePattern : TrivialPattern {
  NonDefaultConstructiblePattern(int);
};

constexpr bool test() {
  { // Check if `base_` and `pattern_` are value initialised
    std::ranges::join_with_view<TrivialView, TrivialPattern> v;
    assert(std::move(v).base().val_ == 0);
    assert(std::ranges::equal(v, std::array{1, 2, 0, 3, 4, 0, 5, 6}));
  }

  { // Default constructor should not be explicit
    [[maybe_unused]] std::ranges::join_with_view<TrivialView, TrivialPattern> v = {};
  }

  static_assert(std::default_initializable<std::ranges::join_with_view<TrivialView, TrivialPattern>>);
  static_assert(!std::default_initializable<std::ranges::join_with_view<TrivialView, NonDefaultConstructiblePattern>>);
  static_assert(!std::default_initializable<std::ranges::join_with_view<NonDefaultConstructibleView, TrivialPattern>>);
  static_assert(!std::default_initializable<
                std::ranges::join_with_view<NonDefaultConstructibleView, NonDefaultConstructiblePattern>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
