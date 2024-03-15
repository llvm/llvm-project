//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <spanstream>

//   template<class charT, class traits = char_traits<charT>>
//   class basic_spanstream
//     : public basic_iostream<charT, traits> {

//     // [spanstream.members], members
//     basic_spanbuf<charT, traits>* rdbuf() const noexcept;

#include <cassert>
#include <span>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_macros.h"

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpStream = std::basic_spanstream<CharT, TraitsT>;

  CharT arr[4];

  std::span<CharT> sp{arr};
  assert(sp.data() == arr);
  assert(!sp.empty());
  assert(sp.size() == 4);

  // Mode: default (`in` | `out`)
  {
    SpStream spSt{sp};
    assert(spSt.rdbuf()->span().data() == arr);
    // Mode `out` counts read characters
    assert(spSt.rdbuf()->span().empty());
    assert(spSt.rdbuf()->span().size() == 0);
  }
  // Mode: `in`
  {
    SpStream spSt{sp, std::ios_base::in};
    assert(spSt.rdbuf()->span().data() == arr);
    assert(!spSt.rdbuf()->span().empty());
    assert(spSt.rdbuf()->span().size() == 4);
  }
  // Mode: `out`
  {
    SpStream spSt{sp, std::ios_base::out};
    assert(spSt.rdbuf()->span().data() == arr);
    // Mode `out` counts read characters
    assert(spSt.rdbuf()->span().empty());
    assert(spSt.rdbuf()->span().size() == 0);
  }
  // Mode: multiple
  {
    SpStream spSt{sp, std::ios_base::in | std::ios_base::out | std::ios_base::binary};
    assert(spSt.rdbuf()->span().data() == arr);
    // Mode `out` counts read characters
    assert(spSt.rdbuf()->span().empty());
    assert(spSt.rdbuf()->span().size() == 0);
  }
  // Mode: `ate`
  {
    SpStream spSt{sp, std::ios_base::out | std::ios_base::ate};
    assert(spSt.rdbuf()->span().data() == arr);
    assert(!spSt.rdbuf()->span().empty());
    assert(spSt.rdbuf()->span().size() == 4);
  }
}

int main(int, char**) {
#ifndef TEST_HAS_NO_NASTY_STRING
  test<nasty_char, nasty_char_traits>();
#endif
  test<char>();
  test<char, constexpr_char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
  test<wchar_t, constexpr_char_traits<wchar_t>>();
#endif

  return 0;
}
