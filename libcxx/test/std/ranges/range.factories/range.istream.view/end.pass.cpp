//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr default_sentinel_t end() const noexcept;

#include <cassert>
#include <ranges>
#include <sstream>

#include "test_macros.h"
#include "utils.h"

template <class T>
concept NoexceptEnd =
    requires(T t) {
      { t.end() } noexcept;
    };

static_assert(NoexceptEnd<std::ranges::istream_view<int>>);
static_assert(NoexceptEnd<const std::ranges::istream_view<int>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(NoexceptEnd<std::ranges::wistream_view<int>>);
static_assert(NoexceptEnd<const std::ranges::wistream_view<int>>);
#endif

template <class CharT>
void test() {
  auto iss = make_string_stream<CharT>("12");
  std::ranges::basic_istream_view<int, CharT> isv{iss};
  [[maybe_unused]] std::same_as<std::default_sentinel_t> auto sent = isv.end();
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
