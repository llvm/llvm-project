//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto begin();

#include <cassert>
#include <ranges>
#include <sstream>

#include "test_macros.h"
#include "utils.h"

template <class T>
concept HasBegin = requires(T t) { t.begin(); };

static_assert(HasBegin<std::ranges::istream_view<int>>);
static_assert(!HasBegin<const std::ranges::istream_view<int>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(HasBegin<std::ranges::wistream_view<int>>);
static_assert(!HasBegin<const std::ranges::wistream_view<int>>);
#endif

template <class CharT>
void test() {
  // begin should read the first element
  {
    auto iss = make_string_stream<CharT>("12    3");
    std::ranges::basic_istream_view<int, CharT> isv{iss};
    auto it = isv.begin();
    assert(*it == 12);
  }

  // empty stream
  {
    auto iss = make_string_stream<CharT>("");
    std::ranges::basic_istream_view<int, CharT> isv{iss};
    auto it = isv.begin();
    assert(it == isv.end());
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
