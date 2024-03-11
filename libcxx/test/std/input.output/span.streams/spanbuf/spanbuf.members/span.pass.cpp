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
//     : public basic_spanbuf<charT, traits> {

//     // [spanbuf.members], member functions
//     std::span<charT> span() const noexcept;

#include <cassert>
#include <span>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "test_convertible.h"
#include "test_macros.h"

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpBuf = std::basic_spanbuf<CharT, TraitsT>;

  CharT arr[4];

  std::span<CharT> sp{arr};
  assert(sp.data() == arr);
  assert(!sp.empty());
  assert(sp.size() == 4);

  // Mode: default (`in` | `out`)
  {
    SpBuf spBuf(sp);
    assert(spBuf.span().data() == arr);
    // Mode `out` counts read characters
    assert(spBuf.span().empty());
    assert(spBuf.span().size() == 0);
  }
  // Mode: `in`
  {
    SpBuf spBuf(sp, std::ios_base::in);
    assert(spBuf.span().data() == arr);
    assert(!spBuf.span().empty());
    assert(spBuf.span().size() == 4);
  }
  // Mode: `out`
  {
    SpBuf spBuf(sp, std::ios_base::out);
    assert(spBuf.span().data() == arr);
    // Mode `out` counts read characters
    assert(spBuf.span().empty());
    assert(spBuf.span().size() == 0);
  }
  // Mode: multiple
  {
    SpBuf spBuf(sp, std::ios_base::in | std::ios_base::out | std::ios_base::binary);
    assert(spBuf.span().data() == arr);
    // Mode `out` counts read characters
    assert(spBuf.span().empty());
    assert(spBuf.span().size() == 0);
  }
  // Mode: `ate`
  {
    SpBuf spBuf(sp, std::ios_base::out | std::ios_base::ate);
    assert(spBuf.span().data() == arr);
    assert(!spBuf.span().empty());
    assert(spBuf.span().size() == 4);
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
