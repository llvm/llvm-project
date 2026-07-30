//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// constexpr explicit iterator(iterator_t<Base> current);

// The constructor is now `private` (exposition-only) per P3059R2.

#include <concepts>
#include <ranges>
#include <tuple>

using BaseIter     = std::tuple<int>*;
using ElementsIter = std::ranges::iterator_t<std::ranges::elements_view<std::ranges::subrange<BaseIter, BaseIter>, 0>>;

static_assert(!std::constructible_from<ElementsIter, BaseIter>);
static_assert(!std::convertible_to<BaseIter, ElementsIter>);
