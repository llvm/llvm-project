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

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpStream = std::basic_ospanstream<CharT, TraitsT>;

  CharT arr[4];
  std::span<CharT> sp{arr};

  // Mode: default (`in` | `out`)
  {
    SpStream rhsSpSt{sp};
    SpStream spSt{std::move(rhsSpSt)};
    assert(spSt.span().data() == arr);
    assert(spSt.span().empty());
    assert(spSt.span().size() == 0);
  }
  // Mode: `in`
  {
    SpStream rhsSpSt{sp, std::ios_base::in};
    SpStream spSt{std::move(rhsSpSt)};
    assert(spSt.span().data() == arr);
    assert(spSt.span().empty());
    assert(spSt.span().size() == 0);
  }
  // Mode `out`
  {
    SpStream rhsSpSt{sp, std::ios_base::out};
    SpStream spSt{std::move(rhsSpSt)};
    assert(spSt.span().data() == arr);
    assert(spSt.span().empty());
    assert(spSt.span().size() == 0);
  }
  // Mode: multiple
  {
    SpStream rhsSpSt{sp, std::ios_base::in | std::ios_base::out | std::ios_base::binary};
    SpStream spSt{std::move(rhsSpSt)};
    assert(spSt.span().data() == arr);
    assert(spSt.span().empty());
    assert(spSt.span().size() == 0);
  }
  // Mode `ate`
  {
    SpStream rhsSpSt{sp, std::ios_base::out | std::ios_base::ate};
    SpStream spSt{std::move(rhsSpSt)};
    assert(spSt.span().data() == arr);
    assert(!spSt.span().empty());
    assert(spSt.span().size() == 4);
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
