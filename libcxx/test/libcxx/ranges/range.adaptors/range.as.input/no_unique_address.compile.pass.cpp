//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// XFAIL: msvc

// <ranges>

// template <input_range _View>
//   requires view<_View>
// class as_input_view : public view_interface<as_input_view<_View>> {
//   _LIBCPP_NO_UNIQUE_ADDRESS _View __base_ = _View();

// Check that members are marked [[no_unique_address]].

#include <ranges>

struct EmptyView : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct NonEmptyView : std::ranges::view_base {
  int* begin() const;
  int* end() const;
  int i;
};

template <class View>
struct Test {
  [[no_unique_address]] View view;
  char c;
};

static_assert(sizeof(std::ranges::as_input_view<EmptyView>) == 1);
static_assert(sizeof(Test<std::ranges::as_input_view<EmptyView>>) == 1);

static_assert(sizeof(std::ranges::as_input_view<NonEmptyView>) == sizeof(NonEmptyView));
