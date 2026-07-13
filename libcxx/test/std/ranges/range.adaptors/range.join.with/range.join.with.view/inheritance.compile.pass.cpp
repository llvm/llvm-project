//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class join_with_view : public view_interface<join_with_view<V, Pattern>>

#include <ranges>

#include <concepts>
#include <string>
#include <vector>

template <class T>
struct View : std::ranges::view_base {
  std::vector<T>* begin();
  std::vector<T>* end();
};

template <class T>
struct Pattern : std::ranges::view_base {
  T* begin();
  T* end();
};

template <class T>
using JoinWithView = std::ranges::join_with_view<View<T>, Pattern<T>>;

static_assert(std::derived_from<JoinWithView<int>, std::ranges::view_interface<JoinWithView<int>>>);
static_assert(std::derived_from<JoinWithView<void*>, std::ranges::view_interface<JoinWithView<void*>>>);
static_assert(std::derived_from<JoinWithView<std::string>, std::ranges::view_interface<JoinWithView<std::string>>>);
