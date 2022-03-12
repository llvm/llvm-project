//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// transform_view() requires std::default_initializable<V> &&
//                           std::default_initializable<F> = default;

#include <ranges>

#include <cassert>
#include <type_traits>

constexpr int buff[] = {1, 2, 3};

struct DefaultConstructibleView : std::ranges::view_base {
  constexpr DefaultConstructibleView() : begin_(buff), end_(buff + 3) { }
  constexpr int const* begin() const { return begin_; }
  constexpr int const* end() const { return end_; }
private:
  int const* begin_;
  int const* end_;
};

struct DefaultConstructibleFunction {
  int state_;
  constexpr DefaultConstructibleFunction() : state_(100) { }
  constexpr int operator()(int i) const { return i + state_; }
};

struct NoDefaultView : std::ranges::view_base {
  NoDefaultView() = delete;
  int* begin() const;
  int* end() const;
};

struct NoDefaultFunction {
  NoDefaultFunction() = delete;
  constexpr int operator()(int i) const;
};

constexpr bool test() {
  {
    std::ranges::transform_view<DefaultConstructibleView, DefaultConstructibleFunction> view;
    assert(view.size() == 3);
    assert(view[0] == 101);
    assert(view[1] == 102);
    assert(view[2] == 103);
  }

  {
    std::ranges::transform_view<DefaultConstructibleView, DefaultConstructibleFunction> view = {};
    assert(view.size() == 3);
    assert(view[0] == 101);
    assert(view[1] == 102);
    assert(view[2] == 103);
  }

  static_assert(!std::is_default_constructible_v<std::ranges::transform_view<NoDefaultView,            DefaultConstructibleFunction>>);
  static_assert(!std::is_default_constructible_v<std::ranges::transform_view<DefaultConstructibleView, NoDefaultFunction>>);
  static_assert(!std::is_default_constructible_v<std::ranges::transform_view<NoDefaultView,            NoDefaultFunction>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
