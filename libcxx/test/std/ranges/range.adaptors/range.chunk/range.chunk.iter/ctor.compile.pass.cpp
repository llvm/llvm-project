//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// V models forward_range:
//   constexpr iterator(iterator<!Const> i)
//     requires Const && convertible_to<iterator_t<V>, iterator_t<Base>> &&
//                  convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <concepts>
#include <ranges>

#include "test_iterators.h"
#include "test_range.h"

using ChunkView     = std::ranges::chunk_view<test_view<forward_iterator>>;
using Iterator      = std::ranges::iterator_t<ChunkView>;
using ConstIterator = std::ranges::iterator_t<const ChunkView>;

// `test_view`'s const and non-const `begin()` return different iterator types, so the converting
// constructor is actually exercised here (unlike a simple_view, where both would collapse into one type).
static_assert(!std::same_as<Iterator, ConstIterator>);
static_assert(std::convertible_to<Iterator, ConstIterator>);
static_assert(!std::convertible_to<ConstIterator, Iterator>);
