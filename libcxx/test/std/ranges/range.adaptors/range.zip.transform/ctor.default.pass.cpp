//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// zip_transform_view() = default;

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

struct NonDefaultConstructibleView : std::ranges::view_base {
  NonDefaultConstructibleView() = delete;
  int* begin() const;
  int* end() const;
};

struct DefaultConstructibleFn {
  constexpr int operator()(const auto&... x) const { return (x + ...); }
};

struct NonDefaultConstructibleFn {
  NonDefaultConstructibleFn() = delete;
  constexpr int operator()(const auto&... x) const;
};

// The default constructor requires all underlying views to be default constructible.
// It is implicitly required by the zip_view's constructor.
static_assert(std::is_default_constructible_v<std::ranges::zip_transform_view< //
                  DefaultConstructibleFn,                                      //
                  DefaultConstructibleView>>);
static_assert(std::is_default_constructible_v<std::ranges::zip_transform_view< //
                  DefaultConstructibleFn,                                      //
                  DefaultConstructibleView,
                  DefaultConstructibleView>>);
static_assert(!std::is_default_constructible_v<std::ranges::zip_transform_view< //
                  NonDefaultConstructibleFn,                                    //
                  DefaultConstructibleView>>);
static_assert(!std::is_default_constructible_v<std::ranges::zip_transform_view< //
                  DefaultConstructibleFn,                                       //
                  NonDefaultConstructibleView>>);
static_assert(!std::is_default_constructible_v<std::ranges::zip_transform_view< //
                  DefaultConstructibleFn,                                       //
                  DefaultConstructibleView,
                  NonDefaultConstructibleView>>);

constexpr bool test() {
  {
    using View =
        std::ranges::zip_transform_view<DefaultConstructibleFn, DefaultConstructibleView, DefaultConstructibleView>;
    View v = View(); // the default constructor is not explicit
    assert(v.size() == 3);
    auto it = v.begin();
    assert(*it++ == 2);
    assert(*it++ == 4);
    assert(*it == 6);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
