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

// CTAD tests.

#include <ranges>
#include <concepts>

#include "test_macros.h"
#include "types.h"

static_assert(std::same_as<decltype(std::ranges::transform_view(InputView(), PlusOne())),
                           std::ranges::transform_view<InputView, PlusOne>>);
static_assert(std::same_as<decltype(std::ranges::transform_view(std::declval<ForwardRange&>(), PlusOne())),
                           std::ranges::transform_view<std::ranges::ref_view<ForwardRange>, PlusOne>>);
static_assert(std::same_as<decltype(std::ranges::transform_view(BorrowableRange(), PlusOne())),
                           std::ranges::transform_view<std::ranges::subrange<int*>, PlusOne>>);
