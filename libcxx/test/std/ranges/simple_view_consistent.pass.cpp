//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// simple_view

#include <ranges>

#include "test_iterators.h"
#include "test_range.h"

struct test_simple_view : std::ranges::view_base {
  constexpr int* begin() const { return nullptr; }
  constexpr int* end() const { return nullptr; }
};

struct test_different_sentinel : std::ranges::view_base {
  constexpr int* begin() const { return nullptr; }
  constexpr int* end() { return nullptr; }
  constexpr sentinel_wrapper<int*> end() const { return sentinel_wrapper<int*>{nullptr}; }
};

struct test_different_begin : std::ranges::view_base {
  constexpr int* begin() { return nullptr; }
  constexpr double* begin() const { return nullptr; }
  constexpr int* end() const { return nullptr; }
};

struct test_non_view {
  constexpr int* begin() const { return nullptr; }
  constexpr int* end() const { return nullptr; }
};

struct test_view_non_const_range : std::ranges::view_base {
  constexpr int* begin() { return nullptr; }
  constexpr int* end() { return nullptr; }
};

static_assert(simple_view<test_simple_view>);
LIBCPP_STATIC_ASSERT(std::ranges::__simple_view<test_simple_view>);

static_assert(!simple_view<test_different_sentinel>);
LIBCPP_STATIC_ASSERT(!std::ranges::__simple_view<test_different_sentinel>);

static_assert(!simple_view<test_different_begin>);
LIBCPP_STATIC_ASSERT(!std::ranges::__simple_view<test_different_begin>);

static_assert(!simple_view<test_non_view>);
LIBCPP_STATIC_ASSERT(!std::ranges::__simple_view<test_non_view>);

static_assert(!simple_view<test_view_non_const_range>);
LIBCPP_STATIC_ASSERT(!std::ranges::__simple_view<test_view_non_const_range>);

int main(int, char**) { return 0; }
