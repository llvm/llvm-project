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
//   class basic_spanbuf
//     : public basic_spanbuf<charT, traits> {

//     // [spanbuf.virtuals], overridden virtual functions
//     basic_streambuf<charT, traits>* setbuf(charT*, streamsize) override;

#include <cassert>
#include <span>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_macros.h"

template <typename CharT, typename TraitsT>
void test() {
  using SpanBuf = std::basic_spanbuf<CharT, TraitsT>;

  CharT arr[4];
  std::span<CharT> sp{arr};

  // Mode: default (`in` | `out`)
  {
    SpanBuf spanBuf{sp};
    assert(spanBuf.span().data() == arr);
    // Mode `out` counts read characters
    assert(spanBuf.span().size() == 0);

    spanBuf.pubsetbuf(nullptr, 0);
    assert(spanBuf.span().data() == nullptr);
    // Mode `out` counts read characters
    assert(spanBuf.span().size() == 0);
  }
  // Mode: `in`
  {
    SpanBuf spanBuf{sp, std::ios_base::in};
    assert(spanBuf.span().data() == arr);
    assert(spanBuf.span().size() == 4);

    spanBuf.pubsetbuf(nullptr, 0);
    assert(spanBuf.span().data() == nullptr);
    assert(spanBuf.span().size() == 0);
  }
  // Mode `out`
  {
    SpanBuf spanBuf{sp, std::ios_base::out};
    assert(spanBuf.span().data() == arr);
    // Mode `out` counts read characters
    assert(spanBuf.span().size() == 0);

    spanBuf.pubsetbuf(nullptr, 0);
    assert(spanBuf.span().data() == nullptr);
    // Mode `out` counts read characters
    assert(spanBuf.span().size() == 0);
  }
  // Mode: `ate`
  {
    SpanBuf spanBuf{sp, std::ios_base::ate};
    assert(spanBuf.span().data() == arr);
    assert(spanBuf.span().size() == 4);

    spanBuf.pubsetbuf(nullptr, 0);
    assert(spanBuf.span().data() == nullptr);
    assert(spanBuf.span().size() == 0);
  }
  // Mode: `ate`
  {
    SpanBuf spanBuf{sp, std::ios_base::out | std::ios_base::ate};
    assert(spanBuf.span().data() == arr);
    assert(spanBuf.span().size() == 4);

    spanBuf.pubsetbuf(nullptr, 0);
    assert(spanBuf.span().data() == nullptr);
    assert(spanBuf.span().size() == 0);
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
