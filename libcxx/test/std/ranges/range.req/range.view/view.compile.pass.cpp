//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <ranges>

// template<class T>
// concept view = ...;

#include <ranges>

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

// const view types

struct ConstView1 : std::ranges::view_base {
  ConstView1(const ConstView1&&);
  const ConstView1& operator=(const ConstView1&&) const;

  friend void swap(const ConstView1&, const ConstView1&);

  friend int* begin(const ConstView1&);
  friend int* end(const ConstView1&);
};
static_assert(std::ranges::range<const ConstView1>);
static_assert(std::movable<const ConstView1>);
static_assert(!std::default_initializable<const ConstView1>);
static_assert(std::ranges::enable_view<const ConstView1>);
static_assert(std::ranges::view<const ConstView1>);

struct ConstView2 : std::ranges::view_interface<ConstView2> {
  ConstView2(const ConstView2&&);
  const ConstView2& operator=(const ConstView2&&) const;

  friend void swap(const ConstView2&, const ConstView2&);

  friend int* begin(const ConstView2&);
  friend int* end(const ConstView2&);
};
static_assert(std::ranges::range<const ConstView2>);
static_assert(std::movable<const ConstView2>);
static_assert(!std::default_initializable<const ConstView2>);
static_assert(std::ranges::enable_view<const ConstView2>);
static_assert(std::ranges::view<const ConstView2>);

// volatile view types
struct VolatileView1 : std::ranges::view_base {
  VolatileView1(volatile VolatileView1&&);
  volatile VolatileView1& operator=(volatile VolatileView1&&) volatile;

  friend void swap(volatile VolatileView1&, volatile VolatileView1&);

  friend int* begin(volatile VolatileView1&);
  friend int* end(volatile VolatileView1&);
};
static_assert(std::ranges::range<volatile VolatileView1>);
static_assert(std::movable<volatile VolatileView1>);
static_assert(!std::default_initializable<volatile VolatileView1>);
static_assert(std::ranges::enable_view<volatile VolatileView1>);
static_assert(std::ranges::view<volatile VolatileView1>);

struct VolatileView2 : std::ranges::view_interface<VolatileView2> {
  VolatileView2(volatile VolatileView2&&);
  volatile VolatileView2& operator=(volatile VolatileView2&&) volatile;

  friend void swap(volatile VolatileView2&, volatile VolatileView2&);

  friend int* begin(volatile VolatileView2&);
  friend int* end(volatile VolatileView2&);
};
static_assert(std::ranges::range<volatile VolatileView2>);
static_assert(std::movable<volatile VolatileView2>);
static_assert(!std::default_initializable<volatile VolatileView2>);
static_assert(std::ranges::enable_view<volatile VolatileView2>);
static_assert(std::ranges::view<volatile VolatileView2>);

// const-volatile view types

struct ConstVolatileView1 : std::ranges::view_base {
  ConstVolatileView1(const volatile ConstVolatileView1&&);
  const volatile ConstVolatileView1& operator=(const volatile ConstVolatileView1&&) const volatile;

  friend void swap(const volatile ConstVolatileView1&, const volatile ConstVolatileView1&);

  friend int* begin(const volatile ConstVolatileView1&);
  friend int* end(const volatile ConstVolatileView1&);
};
static_assert(std::ranges::range<const volatile ConstVolatileView1>);
static_assert(std::movable<const volatile ConstVolatileView1>);
static_assert(!std::default_initializable<const volatile ConstVolatileView1>);
static_assert(std::ranges::enable_view<const volatile ConstVolatileView1>);
static_assert(std::ranges::view<const volatile ConstVolatileView1>);

struct ConstVolatileView2 : std::ranges::view_interface<ConstVolatileView2> {
  ConstVolatileView2(const volatile ConstVolatileView2&&);
  const volatile ConstVolatileView2& operator=(const volatile ConstVolatileView2&&) const volatile;

  friend void swap(const volatile ConstVolatileView2&, const volatile ConstVolatileView2&);

  friend int* begin(const volatile ConstVolatileView2&);
  friend int* end(const volatile ConstVolatileView2&);
};
static_assert(std::ranges::range<const volatile ConstVolatileView2>);
static_assert(std::movable<const volatile ConstVolatileView2>);
static_assert(!std::default_initializable<const volatile ConstVolatileView2>);
static_assert(std::ranges::enable_view<const volatile ConstVolatileView2>);
static_assert(std::ranges::view<const volatile ConstVolatileView2>);
