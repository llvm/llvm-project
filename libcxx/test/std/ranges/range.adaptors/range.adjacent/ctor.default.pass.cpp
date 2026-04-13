//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// adjacent_view() default_initializable<V> = default;

#include <ranges>

#include <cassert>
#include <type_traits>
#include <utility>

constexpr int buff[] = {1, 2, 3, 4, 5};

struct DefaultConstructibleView : std::ranges::view_base {
  constexpr DefaultConstructibleView() : begin_(buff), end_(buff + std::ranges::size(buff)) {}
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

static_assert(std::is_default_constructible_v<std::ranges::adjacent_view<DefaultConstructibleView, 1>>);
static_assert(std::is_default_constructible_v<std::ranges::adjacent_view<DefaultConstructibleView, 2>>);
static_assert(std::is_default_constructible_v<std::ranges::adjacent_view<DefaultConstructibleView, 3>>);
static_assert(!std::is_default_constructible_v<std::ranges::adjacent_view<NoDefaultCtrView, 1>>);
static_assert(!std::is_default_constructible_v<std::ranges::adjacent_view<NoDefaultCtrView, 2>>);
static_assert(!std::is_default_constructible_v<std::ranges::adjacent_view<NoDefaultCtrView, 3>>);

template <std::size_t N>
constexpr void test() {
  {
    using View = std::ranges::adjacent_view<DefaultConstructibleView, N>;
    View v     = View(); // the default constructor is not explicit
    assert(v.size() == std::ranges::size(buff) - (N - 1));
    auto tuple = *v.begin();
    assert(std::get<0>(tuple) == buff[0]);
    if constexpr (N >= 2)
      assert(std::get<1>(tuple) == buff[1]);
    if constexpr (N >= 3)
      assert(std::get<2>(tuple) == buff[2]);
  }
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
