
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <optional>

// template <class U> T optional<T>::value_or(U&&);

#include <concepts>
#include <optional>

template <typename Opt, typename T>
concept has_value_or = requires(Opt opt, T&& t) {
  { opt.value_or(t) } -> std::same_as<T>;
};

static_assert(has_value_or<std::optional<int>, int>);
static_assert(has_value_or<std::optional<int&>, int&>);
static_assert(has_value_or<std::optional<const int&>, const int&>);
static_assert(!has_value_or<std::optional<int (&)[1]>&&, int (&)[1]>);
static_assert(!has_value_or<std::optional<int (&)()>&&, int (&)()>);
