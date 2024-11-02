//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr View base() const& requires copy_constructible<_View>;
// constexpr View base() &&;

#include <ranges>

#include <cassert>
#include <string_view>
#include <utility>
#include "types.h"

struct MoveOnlyView : std::ranges::view_base {
  std::string_view view_;
  constexpr MoveOnlyView() = default;
  constexpr MoveOnlyView(const char* ptr) : view_(ptr) {}
  constexpr MoveOnlyView(std::string_view v) : view_(v) {}
  constexpr MoveOnlyView(MoveOnlyView&&) = default;
  constexpr MoveOnlyView& operator=(MoveOnlyView&&) = default;
  constexpr const char* begin() const { return view_.begin(); }
  constexpr const char* end() const { return view_.end(); }
  constexpr bool operator==(MoveOnlyView rhs) const { return view_ == rhs.view_; }
};
static_assert( std::ranges::view<MoveOnlyView>);
static_assert( std::ranges::contiguous_range<MoveOnlyView>);
static_assert(!std::copyable<MoveOnlyView>);

struct ViewWithInitTracking : std::ranges::view_base {
  enum class InitializedBy {
    Copy,
    Move,
    Invalid
  };

  std::string_view v_;
  InitializedBy initialized_by = InitializedBy::Invalid;
  constexpr ViewWithInitTracking(std::string_view v) : v_(v) {}

  constexpr auto begin() const { return v_.begin(); }
  constexpr auto end() const { return v_.end(); }

  constexpr ViewWithInitTracking(const ViewWithInitTracking& rhs) : v_(rhs.v_) { initialized_by = InitializedBy::Copy; }
  constexpr ViewWithInitTracking(ViewWithInitTracking&& rhs) : v_(rhs.v_) { initialized_by = InitializedBy::Move; }
  constexpr ViewWithInitTracking& operator=(const ViewWithInitTracking& rhs) = default;
  constexpr ViewWithInitTracking& operator=(ViewWithInitTracking&& rhs) = default;
  constexpr bool operator==(const ViewWithInitTracking& rhs) const { return v_ == rhs.v_; }
};

template <class View>
concept CanCallBase = requires(View v) { std::forward<View>(v).base(); };

static_assert( CanCallBase<std::ranges::lazy_split_view<MoveOnlyView, ForwardView>&&>);
static_assert(!CanCallBase<std::ranges::lazy_split_view<MoveOnlyView, ForwardView>&>);
static_assert(!CanCallBase<std::ranges::lazy_split_view<MoveOnlyView, ForwardView> const &>);
static_assert(!CanCallBase<std::ranges::lazy_split_view<MoveOnlyView, ForwardView> const &&>);

constexpr bool test() {
  using View = ViewWithInitTracking;

  // Copyable input -- both lvalue and rvalue overloads of `base` are available.
  {
    // Non-const lvalue.
    {
      View str{"abc def"};
      std::ranges::lazy_split_view<View, std::string_view> v(str, " ");

      std::same_as<View> decltype(auto) result = v.base();
      assert(result == str);
      assert(result.initialized_by == View::InitializedBy::Copy);
    }

    // Const lvalue.
    {
      View str{"abc def"};
      const std::ranges::lazy_split_view<View, std::string_view> v(str, " ");

      std::same_as<View> decltype(auto) result = v.base();
      assert(result == str);
      assert(result.initialized_by == View::InitializedBy::Copy);
    }

    // Non-const rvalue.
    {
      View str{"abc def"};
      std::ranges::lazy_split_view<View, std::string_view> v(str, " ");

      std::same_as<View> decltype(auto) result = std::move(v).base();
      assert(result == str);
      assert(result.initialized_by == View::InitializedBy::Move);
    }

    // Const rvalue.
    {
      View str{"abc def"};
      const std::ranges::lazy_split_view<View, std::string_view> v(str, " ");

      std::same_as<View> decltype(auto) result = std::move(v).base();
      assert(result == str);
      assert(result.initialized_by == View::InitializedBy::Copy);
    }
  }

  // Move-only input -- only the rvalue overload of `base` is available.
  {
    std::ranges::lazy_split_view<MoveOnlyView, ForwardView> v;
    assert(std::move(v).base() == MoveOnlyView());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
