//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <ranges>
#include <utility>
#include <vector>

using JoinView = decltype(std::views::join(std::declval<std::vector<std::vector<int>>&>()));
using JoinIter = std::ranges::iterator_t<JoinView>;
static_assert(std::__is_segmented_iterator<JoinIter>::value);
