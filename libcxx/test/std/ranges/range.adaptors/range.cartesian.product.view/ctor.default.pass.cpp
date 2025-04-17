//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// cartesian_product_view() = default;

#include <ranges>

#include <cassert>
#include <type_traits>
#include <utility>

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

// The default constructor requires all underlying views to be default constructible.
// It is implicitly required by the tuple's constructor. If any of the iterators are
// not default constructible, cartesian product iterator's =default would be implicitly deleted.
static_assert(std::is_default_constructible_v<std::ranges::cartesian_product_view<DefaultConstructibleView>>);
static_assert(std::is_default_constructible_v<
              std::ranges::cartesian_product_view<DefaultConstructibleView, DefaultConstructibleView>>);
static_assert(
    !std::is_default_constructible_v<std::ranges::cartesian_product_view<DefaultConstructibleView, NoDefaultCtrView>>);
static_assert(
    !std::is_default_constructible_v<std::ranges::cartesian_product_view<NoDefaultCtrView, NoDefaultCtrView>>);
static_assert(!std::is_default_constructible_v<std::ranges::cartesian_product_view<NoDefaultCtrView>>);

constexpr bool test() {
  {
    using View = std::ranges::cartesian_product_view<DefaultConstructibleView, DefaultConstructibleView>;
    View v     = View(); // the default constructor is not explicit
    assert(v.size() == 9);
    auto it     = v.begin();
    using Value = std::tuple<const int&, const int&>;
    assert(*it++ == Value(1, 1));
    assert(*it++ == Value(1, 2));
    assert(*it++ == Value(1, 3));
    assert(*it++ == Value(2, 1));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}