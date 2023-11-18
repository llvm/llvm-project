//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <ranges>

#include "test.h"

// template<class T>
// inline constexpr bool enable_borrowed_range<stride_view<T>> = false;

// The stride_view is not one of those range adaptors that (under any circumstances)
// is enabled as a borrowable range by default. In other words, we will have to make
// a positively test case explicity

template <>
inline constexpr bool
    std::ranges::enable_borrowed_range<std::ranges::stride_view<InputView<cpp17_input_iterator<int*>>>> = true;

static_assert(std::ranges::borrowed_range<
              std::ranges::stride_view<InputView<cpp17_input_iterator<int*>>>>);      // expected-no-diagnostics
static_assert(!std::ranges::borrowed_range<InputView<bidirectional_iterator<int*>>>); // expected-no-diagnostics
