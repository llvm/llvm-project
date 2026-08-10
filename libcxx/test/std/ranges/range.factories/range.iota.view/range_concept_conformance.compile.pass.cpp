//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Test that iota_view conforms to range and view concepts.

#include <ranges>

#include "types.h"

struct Decrementable {
  using difference_type = int;

  auto operator<=>(const Decrementable&) const = default;

  Decrementable& operator++();
  Decrementable  operator++(int);
  Decrementable& operator--();
  Decrementable  operator--(int);
};

struct Incrementable {
  using difference_type = int;

  auto operator<=>(const Incrementable&) const = default;

  Incrementable& operator++();
  Incrementable  operator++(int);
};

static_assert(std::ranges::random_access_range<std::ranges::iota_view<int>>);
static_assert(std::ranges::random_access_range<const std::ranges::iota_view<int>>);
static_assert(std::ranges::bidirectional_range<std::ranges::iota_view<Decrementable>>);
static_assert(std::ranges::forward_range<std::ranges::iota_view<Incrementable>>);
static_assert(std::ranges::input_range<std::ranges::iota_view<NotIncrementable>>);
static_assert(std::ranges::view<std::ranges::iota_view<int>>);
