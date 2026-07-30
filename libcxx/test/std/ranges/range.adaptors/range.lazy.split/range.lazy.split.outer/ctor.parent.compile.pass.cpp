//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// explicit std::ranges::lazy_split_view::outer-iterator::outer-iterator(Parent& parent)
//   requires (!forward_range<Base>)

// The constructor is now `private` (exposition-only) per P3059R2.

#include <concepts>
#include <ranges>

#include "../types.h"

static_assert(!std::ranges::forward_range<SplitViewInput>);

static_assert(!std::constructible_from<OuterIterInput, SplitViewInput&>);
static_assert(!std::convertible_to<SplitViewInput&, OuterIterInput>);
