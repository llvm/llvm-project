//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// constexpr explicit inner-iterator::inner-iterator(outer-iterator<Const> i);

// The constructor is now `private` (exposition-only) per P3059R2.

#include <concepts>
#include <ranges>

#include "../types.h"

static_assert(!std::constructible_from<InnerIterConst, OuterIterConst>);
static_assert(!std::convertible_to<InnerIterConst, OuterIterConst>);

static_assert(!std::constructible_from<InnerIterNonConst, OuterIterNonConst>);
static_assert(!std::convertible_to<InnerIterNonConst, OuterIterNonConst>);
