//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// The requirements for transform_view::<iterator>'s members.

#include <ranges>

#include "test_macros.h"
#include "../types.h"

static_assert(std::ranges::bidirectional_range<std::ranges::transform_view<BidirectionalView, IncrementConst>>);
static_assert(!std::ranges::bidirectional_range<std::ranges::transform_view<ForwardView, IncrementConst>>);

static_assert(std::ranges::random_access_range<std::ranges::transform_view<RandomAccessView, IncrementConst>>);
static_assert(!std::ranges::random_access_range<std::ranges::transform_view<BidirectionalView, IncrementConst>>);
