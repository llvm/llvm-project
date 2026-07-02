//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// cartesian_product_view() = default;
//
// The defaulted default constructor requires every underlying view to be default-constructible.

#include <cassert>
#include <ranges>

constexpr int buff[] = {1, 2, 3};

struct DefaultConstructibleView : std::ranges::view_base {
  constexpr DefaultConstructibleView() : begin_(buff), end_(buff + 3) {}
  constexpr int const* begin() const { return begin_; }
  constexpr int const* end() const { return end_; }

private:
  int const* begin_;
  int const* end_;
};

struct NoDefaultCtrView : std::ranges::view_base {
  NoDefaultCtrView() = delete;
  int* begin() const;
  int* end() const;
};

// The default constructor is only available when *every* base view is default-constructible.
static_assert(std::default_initializable<std::ranges::cartesian_product_view<DefaultConstructibleView>>);
static_assert(std::default_initializable<
              std::ranges::cartesian_product_view<DefaultConstructibleView, DefaultConstructibleView>>);
static_assert(std::default_initializable<std::ranges::cartesian_product_view<DefaultConstructibleView,
                                                                             DefaultConstructibleView,
                                                                             DefaultConstructibleView>>);
static_assert(!std::default_initializable<std::ranges::cartesian_product_view<NoDefaultCtrView>>);
static_assert(
    !std::default_initializable<std::ranges::cartesian_product_view<DefaultConstructibleView, NoDefaultCtrView>>);
static_assert(
    !std::default_initializable<std::ranges::cartesian_product_view<NoDefaultCtrView, DefaultConstructibleView>>);

constexpr bool test() {
  { // 2-range default construction iterates correctly
    using View = std::ranges::cartesian_product_view<DefaultConstructibleView, DefaultConstructibleView>;
    View v     = View(); // not explicit
    assert(v.size() == 9);
    auto it     = v.begin();
    using Value = std::tuple<const int&, const int&>;
    assert(*it++ == Value(1, 1));
    assert(*it++ == Value(1, 2));
    assert(*it++ == Value(1, 3));
    assert(*it++ == Value(2, 1));
  }

  { // 3-range default construction
    using View = std::ranges::
        cartesian_product_view<DefaultConstructibleView, DefaultConstructibleView, DefaultConstructibleView>;
    View v = {};
    assert(v.size() == 27);
    auto [a, b, c] = *v.begin();
    assert(a == 1 && b == 1 && c == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
