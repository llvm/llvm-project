//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// std::ranges::as_const_view::as_const_view(...)

#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

struct DefaultConstructibleView : std::ranges::view_base {
  int* begin() const;
  int* end() const;

  int i_ = 23;
};

struct NonDefaultConstructibleView : std::ranges::view_base {
  NonDefaultConstructibleView(int i) : i_(i) {}

  int* begin() const;
  int* end() const;

  int i_ = 23;
};

static_assert(!std::is_constructible_v<std::ranges::as_const_view<NonDefaultConstructibleView>>);
static_assert(std::is_constructible_v<std::ranges::as_const_view<NonDefaultConstructibleView>, int>);
static_assert(std::is_nothrow_constructible_v<std::ranges::as_const_view<DefaultConstructibleView>>);

template <class T, class... Args>
concept IsImplicitlyConstructible = requires(void (&fun)(T), Args&&... args) { fun({std::forward<Args>(args)...}); };

static_assert(IsImplicitlyConstructible<std::ranges::as_const_view<DefaultConstructibleView>>);
static_assert(!IsImplicitlyConstructible<std::ranges::as_const_view<NonDefaultConstructibleView>, int>);

static_assert(std::is_constructible_v<std::ranges::as_const_view<DefaultConstructibleView>, DefaultConstructibleView>);
static_assert(
    std::is_constructible_v<std::ranges::as_const_view<DefaultConstructibleView>, NonDefaultConstructibleView>);
static_assert(!std::is_convertible_v<DefaultConstructibleView, std::ranges::as_const_view<DefaultConstructibleView>>);
static_assert(
    !std::is_convertible_v<DefaultConstructibleView, std::ranges::as_const_view<NonDefaultConstructibleView>>);

constexpr bool test() {
  std::ranges::as_const_view<DefaultConstructibleView> view = {};
  assert(view.base().i_ == 23);

  return true;
}

int main(int, char**) {
  static_assert(test());
  test();

  return 0;
}
