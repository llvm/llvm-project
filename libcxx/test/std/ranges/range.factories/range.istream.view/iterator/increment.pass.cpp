//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: c++03, c++11, c++14, c++17

// iterator& operator++();
// void operator++(int);

#include <cassert>
#include <ranges>
#include <sstream>

#include "test_macros.h"
#include "../utils.h"

template <class CharT>
void test() {
  // operator ++()
  {
    auto iss = make_string_stream<CharT>("1 2 345 ");
    std::ranges::basic_istream_view<int, CharT> isv{iss};
    auto it = isv.begin();
    assert(*it == 1);

    std::same_as<decltype(it)&> decltype(auto) it2 = ++it;
    assert(&it2 == &it);
    assert(*it2 == 2);

    ++it2;
    assert(*it2 == 345);

    ++it2;
    assert(it2 == isv.end());
  }

  // operator ++(int)
  {
    auto iss = make_string_stream<CharT>("1 2 345 ");
    std::ranges::basic_istream_view<int, CharT> isv{iss};
    auto it = isv.begin();
    assert(*it == 1);

    static_assert(std::same_as<decltype(it++), void>);
    it++;
    assert(*it == 2);
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
