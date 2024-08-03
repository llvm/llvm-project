//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class V>
// inline constexpr bool enable_borrowed_range<stride_view<V>> = ranges::enable_borrowed_range<V>;

#include <ranges>
#include <vector>

#include "test_range.h"
#include "types.h"

static_assert(std::ranges::enable_borrowed_range< std::ranges::stride_view<BorrowedView>>);
static_assert(!std::ranges::enable_borrowed_range< std::ranges::stride_view<NonBorrowedView>>);
