//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-has-no-incomplete-format

// <format>

// template<ranges::input_range R>
//   requires same_as<R, remove_cvref_t<R>>
// constexpr range_format format_kind<R> = see below;

#include <format>

#include <array>
#include <queue>
#include <stack>
#include <tuple>
#include <utility>

#include "test_macros.h"

constexpr std::range_format valid = std::format_kind<std::array<int, 1>>;

// expected-error@*:* {{create a template specialization of format_kind for your type}}
constexpr std::range_format invalid_due_to_const = std::format_kind<const std::array<int, 1>>;

// expected-error@*:* {{create a template specialization of format_kind for your type}}
constexpr std::range_format invalid_due_to_volatile = std::format_kind<volatile std::array<int, 1>>;

// expected-error@*:* {{create a template specialization of format_kind for your type}}
constexpr std::range_format invalid_due_to_reference = std::format_kind<std::array<int, 1>&>;

// expected-error@*:* {{create a template specialization of format_kind for your type}}
constexpr std::range_format invalid_no_input_range = std::format_kind<int>;

// expected-error@*:* {{create a template specialization of format_kind for your type}}
constexpr std::range_format not_a_range_stack = std::format_kind<std::stack<int>>;

// expected-error@*:* {{create a template specialization of format_kind for your type}}
constexpr std::range_format not_a_range_queue = std::format_kind<std::queue<int>>;

// expected-error@*:* {{create a template specialization of format_kind for your type}}
constexpr std::range_format not_a_range_priority_queue = std::format_kind<std::priority_queue<int>>;

// expected-error@*:* {{create a template specialization of format_kind for your type}}
constexpr std::range_format not_a_range_pair = std::format_kind<std::pair<int, int>>;

// expected-error@*:* {{create a template specialization of format_kind for your type}}
constexpr std::range_format not_a_range_tuple_1 = std::format_kind<std::tuple<int>>;

// expected-error@*:* {{create a template specialization of format_kind for your type}}
constexpr std::range_format not_a_range_tuple_2 = std::format_kind<std::tuple<int, int>>;

// expected-error@*:* {{create a template specialization of format_kind for your type}}
constexpr std::range_format not_a_range_tuple_3 = std::format_kind<std::tuple<int, int, int>>;
