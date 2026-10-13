//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// std::ranges::lazy_split_view::outer-iterator::value_type

#include <ranges>
#include <type_traits>

#include "../types.h"

// outer-iterator::value_type should not be default-constructible (LWG 4013).
static_assert(!std::is_default_constructible_v<ValueTypeForward>);
static_assert(!std::is_default_constructible_v<ValueTypeInput>);

// outer-iterator::value_type constructor from outer-iterator is private / not constructible by users (LWG 4013).
static_assert(!std::is_constructible_v<ValueTypeForward, OuterIterForward>);
static_assert(!std::is_constructible_v<ValueTypeInput, OuterIterInput>);
static_assert(!std::is_convertible_v<OuterIterForward, ValueTypeForward>);
static_assert(!std::is_convertible_v<OuterIterInput, ValueTypeInput>);
