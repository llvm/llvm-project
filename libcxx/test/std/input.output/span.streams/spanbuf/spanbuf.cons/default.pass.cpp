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
//   class basic_spanbuf
//     : public basic_streambuf<charT, traits> {

//     // [spanbuf.cons], constructors
//     basic_spanbuf();

#include <cassert>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "test_macros.h"

template <typename CharT, typename Traits = std::char_traits<CharT>>
void test() {
  using SpanBuf = std::basic_spanbuf<CharT, Traits>;

  static_assert(std::default_initializable<SpanBuf>);

  SpanBuf spBuf;
  assert(spBuf.span().empty());
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
