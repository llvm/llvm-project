//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// No explicit deduction guides. Only implicit copy deduction
// guide should ever be usable.

// template <class T, size_t N>
//    inplace_vector(inplace_vector<T, N>) -> inplace_vector<T, N>

#include <inplace_vector>
#include <utility>
#include <type_traits>
#include <span>

#include "test_macros.h"
#include "test_iterators.h"

// FIXME: Should _LIBCPP_CTAD_SUPPORTED_FOR_TYPE be added to support
// copy constructors?
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wctad-maybe-unsupported")
TEST_GCC_DIAGNOSTIC_IGNORED("-Wctad-maybe-unsupported")

template <typename T>
constexpr bool copy_deduction() {
  static_assert(std::is_same_v<decltype(std::inplace_vector(std::declval<T>())), T>);
  static_assert(std::is_same_v<decltype(std::inplace_vector(std::declval<const T>())), T>);
  static_assert(std::is_same_v<decltype(std::inplace_vector(std::declval<T&>())), T>);
  static_assert(std::is_same_v<decltype(std::inplace_vector(std::declval<const T&>())), T>);
  static_assert(std::is_same_v<decltype(std::inplace_vector{std::declval<T>()}), T>);
  static_assert(std::is_same_v<decltype(std::inplace_vector{std::declval<const T>()}), T>);
  static_assert(std::is_same_v<decltype(std::inplace_vector{std::declval<T&>()}), T>);
  static_assert(std::is_same_v<decltype(std::inplace_vector{std::declval<const T&>()}), T>);
  return true;
}

static_assert(copy_deduction<std::inplace_vector<int, 10>>());
static_assert(copy_deduction<std::inplace_vector<int, 0>>());

template <class... Args>
concept NotDeducible = !(
    requires(Args... args) { std::inplace_vector(std::forward<Args>(args)...); } ||
    requires(Args... args) { std::inplace_vector{std::forward<Args>(args)...}; });

static_assert(NotDeducible<>);
static_assert(NotDeducible<int>);
static_assert(NotDeducible<std::size_t>);
static_assert(NotDeducible<int*, int*>);
static_assert(NotDeducible<std::from_range_t, int (&)[10]>);
static_assert(NotDeducible<std::from_range_t, std::span<int, 10>>);
static_assert(NotDeducible<std::initializer_list<int>>);

int main(int, char**) { return 0; }
