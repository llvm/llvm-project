//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: c++03, c++11, c++14, c++17

// Val& operator*() const;

#include <cassert>
#include <ranges>
#include <sstream>

#include "test_macros.h"
#include "../utils.h"

template <class CharT>
void test() {
  // operator* should return correct value
  {
    auto iss = make_string_stream<CharT>("1 2 345 ");
    std::ranges::basic_istream_view<int, CharT> isv{iss};
    auto it                              = isv.begin();
    std::same_as<int&> decltype(auto) v1 = *it;
    assert(v1 == 1);
  }

  // operator* should return the same reference to the value stored in the view
  {
    auto iss = make_string_stream<CharT>("1 2 345 ");
    std::ranges::basic_istream_view<int, CharT> isv{iss};
    using Iter = std::ranges::iterator_t<decltype(isv)>;

    Iter it1{isv};
    Iter it2{isv};
    assert(&*it1 == &*it2);
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
