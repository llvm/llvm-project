//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: c++03, c++11, c++14, c++17

// concept checking istream_view

#include <istream>
#include <ranges>

#include "test_macros.h"

template <class Val, class CharT, class Traits = std::char_traits<CharT>>
concept HasIstreamView = requires { typename std::ranges::basic_istream_view<Val, CharT, Traits>; };

static_assert(HasIstreamView<int, char>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(HasIstreamView<int, wchar_t>);
#endif

// Unmovable Val
struct Unmovable {
  Unmovable()            = default;
  Unmovable(Unmovable&&) = delete;
  template <class CharT>
  friend std::basic_istream<CharT>& operator>>(std::basic_istream<CharT>& x, const Unmovable&) {
    return x;
  }
};
static_assert(!HasIstreamView<Unmovable, char>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!HasIstreamView<Unmovable, wchar_t>);
#endif

// !default_initializable<Val>
struct NoDefaultCtor {
  NoDefaultCtor(int) {}
  friend std::istream& operator>>(std::istream& x, const NoDefaultCtor&) { return x; }
};
static_assert(!HasIstreamView<NoDefaultCtor, char>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!HasIstreamView<NoDefaultCtor, wchar_t>);
#endif

// !stream-extractable<Val, CharT, Traits>
struct Foo {};
static_assert(!HasIstreamView<Foo, char>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!HasIstreamView<Foo, wchar_t>);
#endif

template <class T>
concept OnlyInputRange = std::ranges::input_range<T> && !std::ranges::forward_range<T>;

static_assert(OnlyInputRange<std::ranges::istream_view<int>>);
static_assert(OnlyInputRange<std::ranges::istream_view<long>>);
static_assert(OnlyInputRange<std::ranges::istream_view<double>>);
static_assert(OnlyInputRange<std::ranges::istream_view<char>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(OnlyInputRange<std::ranges::wistream_view<int>>);
static_assert(OnlyInputRange<std::ranges::wistream_view<long>>);
static_assert(OnlyInputRange<std::ranges::wistream_view<double>>);
static_assert(OnlyInputRange<std::ranges::wistream_view<wchar_t>>);
#endif
