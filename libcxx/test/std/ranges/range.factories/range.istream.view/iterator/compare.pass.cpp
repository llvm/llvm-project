//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: c++03, c++11, c++14, c++17

// friend bool operator==(const iterator& x, default_sentinel_t);

#include <cassert>
#include <ranges>
#include <sstream>

#include "test_macros.h"
#include "../utils.h"

template <class CharT>
void test() {
  // fail to read
  {
    auto iss = make_string_stream<CharT>("a123 4");
    std::ranges::basic_istream_view<int, CharT> isv{iss};
    auto it = isv.begin();
    assert(it == std::default_sentinel);
  }

  // iterate through the end
  {
    auto iss = make_string_stream<CharT>("123 ");
    std::ranges::basic_istream_view<int, CharT> isv{iss};
    auto it = isv.begin();
    assert(it != std::default_sentinel);
    ++it;
    assert(it == std::default_sentinel);
  }

  // empty stream
  {
    auto iss = make_string_stream<CharT>("");
    std::ranges::basic_istream_view<int, CharT> isv{iss};
    auto it = isv.begin();
    assert(it == std::default_sentinel);
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
