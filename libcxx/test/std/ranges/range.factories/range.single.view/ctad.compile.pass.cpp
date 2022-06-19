//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class T>
//   single_view(T) -> single_view<T>;

#include <ranges>

#include <cassert>
#include <concepts>

#include "test_iterators.h"

struct Empty {};

static_assert(std::same_as<
  decltype(std::ranges::single_view(Empty())),
  std::ranges::single_view<Empty>
>);

static_assert(std::same_as<
  decltype(std::ranges::single_view(std::declval<Empty&>())),
  std::ranges::single_view<Empty>
>);

static_assert(std::same_as<
  decltype(std::ranges::single_view(std::declval<Empty&&>())),
  std::ranges::single_view<Empty>
>);
