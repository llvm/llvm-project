//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

// static constexpr const auto & value = X.data;
// using type = constant_wrapper;
// using value_type = decltype(X)::type;

#include <algorithm>
#include <concepts>
#include <utility>

static_assert(std::constant_wrapper<42>::value == 42);
static_assert(std::same_as<decltype(std::constant_wrapper<42>::value), const int&>);
static_assert(std::same_as<std::constant_wrapper<42>::type, std::constant_wrapper<42>>);
static_assert(std::same_as<std::constant_wrapper<42>::value_type, int>);

struct S {
  int member = 42;
};

static_assert(std::constant_wrapper<S{5}>::value.member == 5);
static_assert(std::same_as<decltype(std::constant_wrapper<S{5}>::value), const S&>);
static_assert(std::same_as<std::constant_wrapper<S{5}>::type, std::constant_wrapper<S{5}>>);
static_assert(std::same_as<std::constant_wrapper<S{5}>::value_type, S>);

static_assert(std::ranges::equal(std::constant_wrapper<"abcd">::value, "abcd"));
static_assert(std::same_as<decltype(std::constant_wrapper<"abcd">::value), const char (&)[5]>);
static_assert(std::same_as<std::constant_wrapper<"abcd">::type, std::constant_wrapper<"abcd">>);
static_assert(std::same_as<std::constant_wrapper<"abcd">::value_type, const char[5]>);
