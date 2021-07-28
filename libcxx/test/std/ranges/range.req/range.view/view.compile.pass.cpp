//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// <ranges>

// template<class T>
// concept view = ...;

#include <ranges>

#include "test_macros.h"

// The type would be a view, but it's not moveable.
struct NotMoveable : std::ranges::view_base {
  NotMoveable() = default;
  NotMoveable(NotMoveable&&) = delete;
  NotMoveable& operator=(NotMoveable&&) = delete;
  friend int* begin(NotMoveable&);
  friend int* begin(NotMoveable const&);
  friend int* end(NotMoveable&);
  friend int* end(NotMoveable const&);
};
static_assert(std::ranges::range<NotMoveable>);
static_assert(!std::movable<NotMoveable>);
static_assert(std::default_initializable<NotMoveable>);
static_assert(std::ranges::enable_view<NotMoveable>);
static_assert(!std::ranges::view<NotMoveable>);

// The type would be a view, but it's not default initializable
struct NotDefaultInit : std::ranges::view_base {
  NotDefaultInit() = delete;
  friend int* begin(NotDefaultInit&);
  friend int* begin(NotDefaultInit const&);
  friend int* end(NotDefaultInit&);
  friend int* end(NotDefaultInit const&);
};
static_assert(std::ranges::range<NotDefaultInit>);
static_assert(std::movable<NotDefaultInit>);
static_assert(!std::default_initializable<NotDefaultInit>);
static_assert(std::ranges::enable_view<NotDefaultInit>);
static_assert(std::ranges::view<NotDefaultInit>);

// The type would be a view, but it doesn't enable it with enable_view
struct NotExplicitlyEnabled {
  NotExplicitlyEnabled() = default;
  NotExplicitlyEnabled(NotExplicitlyEnabled&&) = default;
  NotExplicitlyEnabled& operator=(NotExplicitlyEnabled&&) = default;
  friend int* begin(NotExplicitlyEnabled&);
  friend int* begin(NotExplicitlyEnabled const&);
  friend int* end(NotExplicitlyEnabled&);
  friend int* end(NotExplicitlyEnabled const&);
};
static_assert(std::ranges::range<NotExplicitlyEnabled>);
static_assert(std::movable<NotExplicitlyEnabled>);
static_assert(std::default_initializable<NotExplicitlyEnabled>);
static_assert(!std::ranges::enable_view<NotExplicitlyEnabled>);
static_assert(!std::ranges::view<NotExplicitlyEnabled>);

// The type has everything else, but it's not a range
struct NotARange : std::ranges::view_base {
  NotARange() = default;
  NotARange(NotARange&&) = default;
  NotARange& operator=(NotARange&&) = default;
};
static_assert(!std::ranges::range<NotARange>);
static_assert(std::movable<NotARange>);
static_assert(std::default_initializable<NotARange>);
static_assert(std::ranges::enable_view<NotARange>);
static_assert(!std::ranges::view<NotARange>);

// The type satisfies all requirements
struct View : std::ranges::view_base {
  View() = default;
  View(View&&) = default;
  View& operator=(View&&) = default;
  friend int* begin(View&);
  friend int* begin(View const&);
  friend int* end(View&);
  friend int* end(View const&);
};
static_assert(std::ranges::range<View>);
static_assert(std::movable<View>);
static_assert(std::default_initializable<View>);
static_assert(std::ranges::enable_view<View>);
static_assert(std::ranges::view<View>);
