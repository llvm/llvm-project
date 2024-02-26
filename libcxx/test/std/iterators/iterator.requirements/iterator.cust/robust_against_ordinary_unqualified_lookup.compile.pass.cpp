//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>

// Ordinary unqualified lookup should not be performed.

namespace ns {
struct StructWithGlobalIterCustFunctions {};
} // namespace ns

int&& iter_move(ns::StructWithGlobalIterCustFunctions);
void iter_swap(ns::StructWithGlobalIterCustFunctions, ns::StructWithGlobalIterCustFunctions);

#include <iterator>
#include <type_traits>

static_assert(!std::is_invocable_v<decltype(std::ranges::iter_move), ns::StructWithGlobalIterCustFunctions&>);
static_assert(!std::is_invocable_v<decltype(std::ranges::iter_swap),
                                   ns::StructWithGlobalIterCustFunctions&,
                                   ns::StructWithGlobalIterCustFunctions&>);
