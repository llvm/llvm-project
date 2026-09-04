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

#include <ranges>
#include <type_traits>

#include "../types.h"

static_assert(!std::is_constructible_v<InnerIterConst, OuterIterConst>);
static_assert(!std::is_convertible_v<InnerIterConst, OuterIterConst>);

static_assert(!std::is_constructible_v<InnerIterNonConst, OuterIterNonConst>);
static_assert(!std::is_convertible_v<InnerIterNonConst, OuterIterNonConst>);
