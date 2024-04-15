//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class T, class Bound>
//    repeat_view(T, Bound) -> repeat_view<T, Bound>;

#include <concepts>
#include <ranges>
#include <utility>

struct Empty {};

// clang-format off
static_assert(std::same_as<decltype(std::ranges::repeat_view(Empty())), std::ranges::repeat_view<Empty>>);
static_assert(std::same_as<decltype(std::ranges::repeat_view(std::declval<Empty&>())), std::ranges::repeat_view<Empty>>);
static_assert(std::same_as<decltype(std::ranges::repeat_view(std::declval<Empty&&>())), std::ranges::repeat_view<Empty>>);
static_assert(std::same_as<decltype(std::ranges::repeat_view(10, 1)), std::ranges::repeat_view<int, int>>);
static_assert(std::same_as<decltype(std::ranges::repeat_view(10, 1U)), std::ranges::repeat_view<int, unsigned>>);
static_assert(std::same_as<decltype(std::ranges::repeat_view(10, 1UL)), std::ranges::repeat_view<int, unsigned long>>);

// LWG4053 and LWG4054 "Repeating a repeat_view should repeat the view"
static_assert(std::same_as<decltype(std::ranges::repeat_view(std::ranges::repeat_view(1))), std::ranges::repeat_view<std::ranges::repeat_view<int>>>);
// clang-format on
