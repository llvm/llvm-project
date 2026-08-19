//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models forward_range:
//     constexpr iterator(iterator<!Const> i)
//       requires Const && convertible_to<iterator_t<V>, iterator_t<Base>> &&
//                    convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <concepts>
#include <ranges>

#include "test_iterators.h"
#include "test_range.h"

// `test_view` is not a `simple_view`.

static_assert(!std::same_as< std::ranges::iterator_t< std::ranges::chunk_view<test_view<forward_iterator>>>,
                             std::ranges::iterator_t<const std::ranges::chunk_view<test_view<forward_iterator>>> >);
static_assert(std::convertible_to<std::ranges::iterator_t<std::ranges::chunk_view<test_view<forward_iterator>>>,
                                  std::ranges::iterator_t<const std::ranges::chunk_view<test_view<forward_iterator>>>>);
static_assert(!std::convertible_to<std::ranges::iterator_t<const std::ranges::chunk_view<test_view<forward_iterator>>>,
                                   std::ranges::iterator_t<std::ranges::chunk_view<test_view<forward_iterator>>>>);
