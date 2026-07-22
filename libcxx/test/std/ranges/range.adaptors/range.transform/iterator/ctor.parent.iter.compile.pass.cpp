//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

//  constexpr iterator(Parent& parent, iterator_t<Base> current);

// The constructor is now `private` (exposition-only) per P3059R2.

#include <ranges>

#include "../types.h"

using TransformView = std::ranges::transform_view<MoveOnlyView, PlusOne>;
using TransformViewBaseIter =  std::ranges::iterator_t<MoveOnlyView>;
using TransformIter = std::ranges::iterator_t<TransformView>;

static_assert(!std::is_constructible_v<TransformIter, TransformView, TransformViewBaseIter>);
