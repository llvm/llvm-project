//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr explicit repeat_view(const T& value, Bound bound = Bound()) requires copy_constructible<T>;
// constexpr explicit repeat_view(T&& value, Bound bound = Bound());

#include <ranges>
#include <cassert>
#include <iterator>
#include <type_traits>

#include "MoveOnly.h"

struct Empty {};

// Test explicit
static_assert(std::is_constructible_v<std::ranges::repeat_view<Empty>, const Empty&>);
static_assert(std::is_constructible_v<std::ranges::repeat_view<Empty>, Empty&&>);
static_assert(std::is_constructible_v<std::ranges::repeat_view<Empty, int>, const Empty&>);
static_assert(std::is_constructible_v<std::ranges::repeat_view<Empty, int>, Empty&&>);

static_assert(!std::is_convertible_v<const Empty&, std::ranges::repeat_view<Empty>>);
static_assert(!std::is_convertible_v<Empty&&, std::ranges::repeat_view<Empty>>);
static_assert(!std::is_convertible_v<const Empty&, std::ranges::repeat_view<Empty, int>>);
static_assert(!std::is_convertible_v<Empty&&, std::ranges::repeat_view<Empty, int>>);

static_assert(!std::is_constructible_v<std::ranges::repeat_view<MoveOnly>, const MoveOnly&>);
static_assert(std::is_constructible_v<std::ranges::repeat_view<MoveOnly>, MoveOnly&&>);

constexpr bool test() {
  // Move && unbound && default argument
  {
    std::ranges::repeat_view<Empty> rv(Empty{});
    assert(rv.begin() + 10 != rv.end());
  }

  // Move && unbound && user-provided argument
  {
    std::ranges::repeat_view<Empty> rv(Empty{}, std::unreachable_sentinel);
    assert(rv.begin() + 10 != rv.end());
  }

  // Move && bound && default argument
  {
    std::ranges::repeat_view<Empty, int> rv(Empty{});
    assert(rv.begin() == rv.end());
  }

  // Move && bound && user-provided argument
  {
    std::ranges::repeat_view<Empty, int> rv(Empty{}, 10);
    assert(rv.begin() + 10 == rv.end());
  }

  // Copy && unbound && default argument
  {
    Empty e;
    std::ranges::repeat_view<Empty> rv(e);
    assert(rv.begin() + 10 != rv.end());
  }

  // Copy && unbound && user-provided argument
  {
    Empty e;
    std::ranges::repeat_view<Empty> rv(e, std::unreachable_sentinel);
    assert(rv.begin() + 10 != rv.end());
  }

  // Copy && bound && default argument
  {
    Empty e;
    std::ranges::repeat_view<Empty, int> rv(e);
    assert(rv.begin() == rv.end());
  }

  // Copy && bound && user-provided argument
  {
    Empty e;
    std::ranges::repeat_view<Empty, int> rv(e, 10);
    assert(rv.begin() + 10 == rv.end());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
