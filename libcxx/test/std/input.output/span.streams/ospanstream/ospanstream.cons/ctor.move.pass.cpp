//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <spanstream>

//   template<class charT, class traits = char_traits<charT>>
//   class basic_ospanstream
//     : public basic_ostream<charT, traits> {

//     // [spanstream.cons], constructors

//     basic_ospanstream(basic_ospanstream&& rhs);

#include <cassert>
#include <span>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_macros.h"

template <typename CharT, typename TraitsT>
void test() {
  using SpStream = std::basic_ospanstream<CharT, TraitsT>;

  CharT arr[4];

  std::span<CharT> sp{arr};
  assert(sp.data() == arr);
  assert(sp.size() == 4);

  // Mode: default (`out`)
  {
    SpStream rhsSpSt{sp};
    assert(rhsSpSt.span().data() == arr);
    // Mode `out` counts read characters
    assert(rhsSpSt.span().size() == 0);

    SpStream spSt{std::move(rhsSpSt)};
    assert(spSt.span().data() == arr);
    // Mode `out` counts read characters
    assert(spSt.span().size() == 0);

    // Test after move
    assert(rhsSpSt.span().data() == arr);
    assert(rhsSpSt.span().size() == 0);
  }
  // Mode `out`
  {
    SpStream rhsSpSt{sp, std::ios_base::out};
    assert(rhsSpSt.span().data() == arr);
    // Mode `out` counts read characters
    assert(rhsSpSt.span().size() == 0);

    SpStream spSt{std::move(rhsSpSt)};
    assert(spSt.span().data() == arr);
    // Mode `out` counts read characters
    assert(spSt.span().size() == 0);

    // Test after move
    assert(rhsSpSt.span().data() == arr);
    assert(rhsSpSt.span().size() == 0);
  }
  // Mode `ate`
  {
    SpStream rhsSpSt{sp, std::ios_base::ate};
    assert(rhsSpSt.span().data() == arr);
    assert(rhsSpSt.span().size() == 4);

    SpStream spSt{std::move(rhsSpSt)};
    assert(spSt.span().data() == arr);
    assert(spSt.span().size() == 4);

    // Test after move
    assert(rhsSpSt.span().data() == arr);
    assert(rhsSpSt.span().size() == 4);
  }
}

int main(int, char**) {
#ifndef TEST_HAS_NO_NASTY_STRING
  test<nasty_char, nasty_char_traits>();
#endif

  test<char, constexpr_char_traits<char>>();
  test<char, std::char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t, constexpr_char_traits<wchar_t>>();
  test<wchar_t, std::char_traits<wchar_t>>();
#endif

  return 0;
}
