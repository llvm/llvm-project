//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <string_view>

// basic_string_view(nullptr_t) = delete; // C++2b

#include <string_view>
#include <type_traits>

static_assert(!std::is_convertible_v<decltype(nullptr), std::string_view>);
static_assert(!std::is_constructible_v<std::string_view, decltype(nullptr)>);
static_assert(!std::is_assignable_v<std::string_view, decltype(nullptr)>);
