//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// constexpr explicit iterator(basic_istream_view& parent) noexcept;

// The constructor is now `private` (exposition-only) per P3059R2.

#include <ranges>
#include <sstream>
#include <type_traits>

#include "test_macros.h"

// test that the constructor is explicit
template <class CharT>
using IstreamView = std::ranges::basic_istream_view<int, CharT>;
template <class CharT>
using Iter = std::ranges::iterator_t<IstreamView<CharT>>;

static_assert(!std::constructible_from<Iter<char>, IstreamView<char>&>);
static_assert(!std::convertible_to<IstreamView<char>&, Iter<char>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::constructible_from<Iter<wchar_t>, IstreamView<wchar_t>&>);
static_assert(!std::convertible_to<IstreamView<wchar_t>&, Iter<wchar_t>>);
#endif
