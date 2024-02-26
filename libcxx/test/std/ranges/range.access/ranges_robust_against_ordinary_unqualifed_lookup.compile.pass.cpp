//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <ranges>

// Ordinary unqualified lookup should not be performed.

namespace ns {
struct StructWithGlobalRangeAccessFunctions {};
} // namespace ns

int* begin(ns::StructWithGlobalRangeAccessFunctions);
int* end(ns::StructWithGlobalRangeAccessFunctions);
int* rbegin(ns::StructWithGlobalRangeAccessFunctions);
int* rend(ns::StructWithGlobalRangeAccessFunctions);
unsigned int size(ns::StructWithGlobalRangeAccessFunctions);

#include <ranges>
#include <type_traits>

static_assert(!std::is_invocable_v<decltype(std::ranges::begin), ns::StructWithGlobalRangeAccessFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::cbegin), ns::StructWithGlobalRangeAccessFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::end), ns::StructWithGlobalRangeAccessFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::cend), ns::StructWithGlobalRangeAccessFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::rbegin), ns::StructWithGlobalRangeAccessFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::crbegin), ns::StructWithGlobalRangeAccessFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::rend), ns::StructWithGlobalRangeAccessFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::crend), ns::StructWithGlobalRangeAccessFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::size), ns::StructWithGlobalRangeAccessFunctions&>);
