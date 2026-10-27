//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>

// text_encoding::aliases_view;

#include <concepts>
#include <text_encoding>

static_assert(std::copyable<std::text_encoding::aliases_view>);
static_assert(std::ranges::view<std::text_encoding::aliases_view>);
static_assert(std::ranges::random_access_range<std::text_encoding::aliases_view>);
static_assert(std::ranges::borrowed_range<std::text_encoding::aliases_view>);
static_assert(std::same_as<std::ranges::range_value_t<std::text_encoding::aliases_view>, const char*>);
static_assert(std::same_as<std::ranges::range_reference_t<std::text_encoding::aliases_view>, const char*>);
