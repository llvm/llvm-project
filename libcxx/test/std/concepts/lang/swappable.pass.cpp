//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T>
// concept swappable = // see below

#include <concepts>

#include <algorithm>
#include <cassert>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <optional>
#include <unordered_map>
#include <vector>

#include "test_macros.h"
#include "moveconstructible.h"
#include "swappable.h"

template <class T>
struct expected {
  T x;
  T y;
};

// clang-format off
// Checks [concept.swappable]/2.1
template <class T, class U>
requires std::same_as<std::remove_cvref_t<T>, std::remove_cvref_t<U> > &&
         std::swappable<std::remove_cvref_t<T> >
constexpr bool check_swap_21(T&& x, U&& y) {
  expected<std::remove_cvref_t<T> > const e{y, x};
  std::ranges::swap(std::forward<T>(x), std::forward<U>(y));
  return x == e.x && y == e.y;
}

// Checks [concept.swappable]/2.2
template <std::swappable T, std::size_t N>
constexpr bool check_swap_22(T (&x)[N], T (&y)[N]) {
  expected<T[N]> e;
  std::copy(y, y + N, e.x);
  std::copy(x, x + N, e.y);

  std::ranges::swap(x, y);
  return std::equal(x, x + N, e.x, e.x + N) &&
         std::equal(y, y + N, e.y, e.y + N);
}

// Checks [concept.swappable]/2.3
template <std::swappable T>
requires std::copy_constructible<std::remove_cvref_t<T> >
constexpr bool check_swap_23(T x, T y) {
  expected<std::remove_cvref_t<T> > const e{y, x};
  std::ranges::swap(x, y);
  return x == e.x && y == e.y;
}
// clang-format on

constexpr bool check_lvalue_adl_swappable() {
  auto x = lvalue_adl_swappable(0);
  auto y = lvalue_adl_swappable(1);
  constexpr auto is_noexcept = noexcept(std::ranges::swap(x, y));
  return check_swap_21(x, y) && is_noexcept;
}
static_assert(check_lvalue_adl_swappable());

constexpr bool check_rvalue_adl_swappable() {
  constexpr auto is_noexcept = noexcept(
      std::ranges::swap(rvalue_adl_swappable(0), rvalue_adl_swappable(1)));
  return check_swap_21(rvalue_adl_swappable(0), rvalue_adl_swappable(1)) &&
         is_noexcept;
}
static_assert(check_rvalue_adl_swappable());

constexpr bool check_lvalue_rvalue_adl_swappable() {
  auto x = lvalue_rvalue_adl_swappable(0);
  constexpr auto is_noexcept =
      noexcept(std::ranges::swap(x, lvalue_rvalue_adl_swappable(1)));
  return check_swap_21(x, lvalue_rvalue_adl_swappable(1)) && is_noexcept;
}
static_assert(check_lvalue_rvalue_adl_swappable());

constexpr bool check_rvalue_lvalue_adl_swappable() {
  auto x = rvalue_lvalue_adl_swappable(0);
  constexpr auto is_noexcept =
      noexcept(std::ranges::swap(rvalue_lvalue_adl_swappable(1), x));
  return check_swap_21(rvalue_lvalue_adl_swappable(1), x) && is_noexcept;
}
static_assert(check_rvalue_lvalue_adl_swappable());

constexpr bool check_throwable_swappable() {
  auto x = throwable_adl_swappable{0};
  auto y = throwable_adl_swappable{1};
  constexpr auto not_noexcept = !noexcept(std::ranges::swap(x, y));
  return check_swap_21(x, y) && not_noexcept;
}
static_assert(check_throwable_swappable());

constexpr bool check_non_move_constructible_adl_swappable() {
  auto x = non_move_constructible_adl_swappable{0};
  auto y = non_move_constructible_adl_swappable{1};
  constexpr auto is_noexcept = noexcept(std::ranges::swap(x, y));
  return check_swap_21(x, y) && is_noexcept;
}
static_assert(check_non_move_constructible_adl_swappable());

constexpr bool check_non_move_assignable_adl_swappable() {
  auto x = non_move_assignable_adl_swappable{0};
  auto y = non_move_assignable_adl_swappable{1};
  return check_swap_21(x, y) && noexcept(std::ranges::swap(x, y));
}
static_assert(check_non_move_assignable_adl_swappable());

namespace swappable_namespace {
enum unscoped { hello, world };
void swap(unscoped&, unscoped&);

enum class scoped { hello, world };
void swap(scoped&, scoped&);
} // namespace swappable_namespace

static_assert(std::swappable<swappable_namespace::unscoped>);
static_assert(std::swappable<swappable_namespace::scoped>);

constexpr bool check_swap_arrays() {
  int x[] = {0, 1, 2, 3, 4};
  int y[] = {5, 6, 7, 8, 9};
  return check_swap_22(x, y) && noexcept(std::ranges::swap(x, y));
}
static_assert(check_swap_arrays());

constexpr bool check_lvalue_adl_swappable_arrays() {
  lvalue_adl_swappable x[] = {{0}, {1}, {2}, {3}};
  lvalue_adl_swappable y[] = {{4}, {5}, {6}, {7}};
  return check_swap_22(x, y) && noexcept(std::ranges::swap(x, y));
}
static_assert(check_lvalue_adl_swappable_arrays());

constexpr bool check_throwable_adl_swappable_arrays() {
  throwable_adl_swappable x[] = {{0}, {1}, {2}, {3}};
  throwable_adl_swappable y[] = {{4}, {5}, {6}, {7}};
  return check_swap_22(x, y) && !noexcept(std::ranges::swap(x, y));
}
static_assert(check_throwable_adl_swappable_arrays());

inline auto global_x = 0;
static_assert(check_swap_23(0, 0) &&
              noexcept(std::ranges::swap(global_x, global_x)));
static_assert(check_swap_23(0, 1) &&
              noexcept(std::ranges::swap(global_x, global_x)));
static_assert(check_swap_23(1, 0) &&
              noexcept(std::ranges::swap(global_x, global_x)));

constexpr bool check_swappable_references() {
  int x = 42;
  int y = 64;
  return check_swap_23<int&>(x, y) && noexcept(std::ranges::swap(x, y));
}
static_assert(check_swappable_references());

constexpr bool check_swappable_pointers() {
  char const* x = "hello";
  return check_swap_23<char const*>(x, nullptr) &&
         noexcept(std::ranges::swap(x, x));
}
static_assert(check_swappable_pointers());

namespace union_swap {
union adl_swappable {
  int x;
  double y;
};

void swap(adl_swappable&, adl_swappable&);
void swap(adl_swappable&&, adl_swappable&&);
}; // namespace union_swap
static_assert(std::swappable<union_swap::adl_swappable>);
static_assert(std::swappable<union_swap::adl_swappable&>);
static_assert(std::swappable<union_swap::adl_swappable&&>);

// All tests for std::swappable<T> are implicitly confirmed by `check_swap`, so we only need to
// sanity check for a few positive cases.
static_assert(std::swappable<int volatile&>);
static_assert(std::swappable<int&&>);
static_assert(std::swappable<int (*)()>);
static_assert(std::swappable<int rvalue_adl_swappable::*>);
static_assert(std::swappable<int (rvalue_adl_swappable::*)()>);
static_assert(std::swappable<std::unique_ptr<int> >);

static_assert(!std::swappable<void>);
static_assert(!std::swappable<int const>);
static_assert(!std::swappable<int const&>);
static_assert(!std::swappable<int const&&>);
static_assert(!std::swappable<int const volatile>);
static_assert(!std::swappable<int const volatile&>);
static_assert(!std::swappable<int const volatile&&>);
static_assert(!std::swappable<int (&)()>);
static_assert(!std::swappable<DeletedMoveCtor>);
static_assert(!std::swappable<ImplicitlyDeletedMoveCtor>);
static_assert(!std::swappable<DeletedMoveAssign>);
static_assert(!std::swappable<ImplicitlyDeletedMoveAssign>);
static_assert(!std::swappable<NonMovable>);
static_assert(!std::swappable<DerivedFromNonMovable>);
static_assert(!std::swappable<HasANonMovable>);

using swap_type = std::remove_const_t<decltype(std::ranges::swap)>;
static_assert(std::default_initializable<swap_type>);
static_assert(std::move_constructible<swap_type>);
static_assert(std::copy_constructible<swap_type>);
static_assert(std::assignable_from<swap_type&, swap_type>);
static_assert(std::assignable_from<swap_type&, swap_type&>);
static_assert(std::assignable_from<swap_type&, swap_type const&>);
static_assert(std::assignable_from<swap_type&, swap_type const>);
static_assert(std::swappable<swap_type>);

template <bool is_noexcept, std::swappable T>
void check_swap(expected<T> const& e) {
  auto a = e.y;
  auto b = e.x;

  std::ranges::swap(a, b);
  assert(a == e.x);
  assert(b == e.y);

  std::ranges::swap(a, b);
  assert(a == e.y);
  assert(b == e.x);

  static_assert(noexcept(std::ranges::swap(a, b)) == is_noexcept);
}

int main(int, char**) {
  {
    auto const e = expected<std::deque<int> >{
        .x = {6, 7, 8, 9},
        .y = {0, 1, 2, 3, 4, 5},
    };
    check_swap</*is_noexcept=*/true>(e);
  }
  {
    auto const e = expected<std::map<int, std::string> >{
        .x = {{0, "whole"}, {1, "cashews"}},
        .y = {{-1, "roasted"}, {2, "&"}, {-3, "salted"}},
    };
    check_swap</*is_noexcept=*/true>(e);
  }
  {
    auto const e = expected<std::string>{
        .x = "hello there",
        .y = "general kenobi",
    };
    check_swap</*is_noexcept=*/true>(e);
  }
  {
    auto const e = expected<std::optional<lvalue_adl_swappable> >{
        .x = {10},
        .y = {20},
    };
    check_swap</*is_noexcept=*/true>(e);
  }
  {
    auto const e = expected<std::optional<throwable_adl_swappable> >{
        .x = {10},
        .y = {20},
    };
    check_swap</*is_noexcept=*/false>(e);
  }
  {
    auto const e = expected<std::unordered_map<int, std::string> >{
        .x = {{0, "whole"}, {1, "cashews"}},
        .y = {{-1, "roasted"}, {2, "&"}, {-3, "salted"}},
    };
    check_swap</*is_noexcept=*/true>(e);
  }
  {
    auto const e = expected<std::vector<int> >{
        .x = {0, 1, 2, 3, 4, 5},
        .y = {6, 7, 8, 9},
    };

    check_swap</*is_noexcept=*/true>(e);
  }
  return 0;
}
