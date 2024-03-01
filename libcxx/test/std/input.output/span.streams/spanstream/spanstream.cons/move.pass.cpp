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
//     : public basic_streambuf<charT, traits> {

//     // [spanbuf.cons], constructors
//
//     basic_spanstream(basic_spanstream&& rhs);

#include <cassert>
#include <concepts>
#include <span>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "test_convertible.h"
#include "test_macros.h"

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpStream = std::basic_spanstream<CharT, TraitsT>;

  CharT arr[4];
  std::span<CharT> sp{arr};

  // Mode: default
  {
    SpStream rhsSpStream{sp};
    SpStream sps(std::move(rhsSpStream));
    assert(sps.span().data() == arr);
    assert(sps.span().empty());
    assert(sps.span().size() == 0);
  }
  // Mode: `ios_base::in`
  {
    SpStream rhsSpStream{sp, std::ios_base::in};
    SpStream sps(std::move(rhsSpStream));
    assert(sps.span().data() == arr);
    assert(!sps.span().empty());
    assert(sps.span().size() == 4);
  }
  // Mode `ios_base::out`
  {
    SpStream rhsSpStream{sp, std::ios_base::out};
    SpStream sps(std::move(rhsSpStream));
    assert(sps.span().data() == arr);
    assert(sps.span().empty());
    assert(sps.span().size() == 0);
  }
  // Mode: multiple
  {
    SpStream rhsSpStream{sp, std::ios_base::in | std::ios_base::out | std::ios_base::binary};
    SpStream sps(std::move(rhsSpStream));
    assert(sps.span().data() == arr);
    assert(sps.span().empty());
    assert(sps.span().size() == 0);
  }
}

int main(int, char**) {
  test<char>();
  test<char, constexpr_char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
  test<wchar_t, constexpr_char_traits<wchar_t>>();
#endif

  return 0;
}
