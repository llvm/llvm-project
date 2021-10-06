//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class R>
// drop_view(R&&, range_difference_t<R>) -> drop_view<views::all_t<R>>;

#include <ranges>

#include "test_macros.h"
#include "types.h"

namespace ranges = std::ranges;

static_assert(std::same_as<decltype(ranges::drop_view(MoveOnlyView(), 0)), ranges::drop_view<MoveOnlyView>>);
static_assert(std::same_as<decltype(ranges::drop_view(CopyableView(), 0)), ranges::drop_view<CopyableView>>);
static_assert(std::same_as<decltype(ranges::drop_view(ForwardView(), 0)), ranges::drop_view<ForwardView>>);
static_assert(std::same_as<decltype(ranges::drop_view(InputView(), 0)), ranges::drop_view<InputView>>);

static_assert(std::same_as<decltype(ranges::drop_view(std::declval<ForwardRange&>(), 0)),
                           ranges::drop_view<ranges::ref_view<ForwardRange>>>);

static_assert(std::same_as<decltype(ranges::drop_view(BorrowableRange(), 0)),
                           ranges::drop_view<ranges::subrange<int*>>>);
