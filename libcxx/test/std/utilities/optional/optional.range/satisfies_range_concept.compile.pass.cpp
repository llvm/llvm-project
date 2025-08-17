//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: std-at-least-c++26

#include <optional>
#include <ranges>

static_assert(std::ranges::sized_range<std::optional<int>>);
static_assert(std::ranges::common_range<std::optional<int>>);
static_assert(std::ranges::contiguous_range<std::optional<int>>);
