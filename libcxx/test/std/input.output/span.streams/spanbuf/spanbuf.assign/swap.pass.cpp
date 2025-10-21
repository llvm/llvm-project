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

//     // [ispanstream.swap], swap
//     void swap(basic_spanbuf& rhs);

#include <cassert>
#include <span>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_macros.h"

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpanBuf = std::basic_spanbuf<CharT, TraitsT>;

  CharT arr[4];

  std::span<CharT> sp{arr};
  assert(sp.data() == arr);
  assert(sp.size() == 4);

  // Mode: default (`in` | `out`)
  {
    SpanBuf rhsSpanBuf{sp};
    assert(rhsSpanBuf.span().data() == arr);
    // Mode `out` counts read characters
    assert(rhsSpanBuf.span().size() == 0);

    SpanBuf spanBuf;
    assert(spanBuf.span().data() == nullptr);
    // Mode `out` counts read characters
    assert(spanBuf.span().size() == 0);

    spanBuf.swap(rhsSpanBuf);
    assert(spanBuf.span().data() == arr);
    assert(spanBuf.span().size() == 0);
    assert(rhsSpanBuf.span().data() == nullptr);
    assert(rhsSpanBuf.span().size() == 0);
  }
  // Mode: `in`
  {
    SpanBuf rhsSpanBuf{sp, std::ios_base::in};
    assert(rhsSpanBuf.span().data() == arr);
    assert(rhsSpanBuf.span().size() == 4);

    SpanBuf spanBuf(std::span<CharT>{});
    assert(spanBuf.span().data() == nullptr);
    assert(spanBuf.span().size() == 0);

    spanBuf.swap(rhsSpanBuf);
    assert(spanBuf.span().data() == arr);
    assert(spanBuf.span().size() == 4);
    assert(rhsSpanBuf.span().data() == nullptr);
    assert(rhsSpanBuf.span().size() == 0);
  }
  // Mode `out`
  {
    SpanBuf rhsSpanBuf{sp, std::ios_base::out};
    assert(rhsSpanBuf.span().data() == arr);
    // Mode `out` counts read characters
    assert(rhsSpanBuf.span().size() == 0);

    SpanBuf spanBuf;
    assert(spanBuf.span().data() == nullptr);
    // Mode `out` counts read characters
    assert(spanBuf.span().size() == 0);

    spanBuf.swap(rhsSpanBuf);
    assert(spanBuf.span().data() == arr);
    assert(spanBuf.span().size() == 0);
    assert(rhsSpanBuf.span().data() == nullptr);
    assert(rhsSpanBuf.span().size() == 0);
  }
  // Mode: `ate`
  {
    SpanBuf rhsSpanBuf{sp, std::ios_base::ate};
    assert(rhsSpanBuf.span().data() == arr);
    assert(rhsSpanBuf.span().size() == 4);

    SpanBuf spanBuf(std::span<CharT>{});
    assert(spanBuf.span().data() == nullptr);
    assert(spanBuf.span().size() == 0);

    spanBuf.swap(rhsSpanBuf);
    assert(spanBuf.span().data() == arr);
    assert(spanBuf.span().size() == 4);
    assert(rhsSpanBuf.span().data() == nullptr);
    assert(rhsSpanBuf.span().size() == 0);
  }
  // Mode: `ate`
  {
    SpanBuf rhsSpanBuf{sp, std::ios_base::out | std::ios_base::ate};
    assert(rhsSpanBuf.span().data() == arr);
    assert(rhsSpanBuf.span().size() == 4);

    SpanBuf spanBuf(std::span<CharT>{});
    assert(spanBuf.span().data() == nullptr);
    assert(spanBuf.span().size() == 0);

    spanBuf.swap(rhsSpanBuf);
    assert(spanBuf.span().data() == arr);
    assert(spanBuf.span().size() == 4);
    assert(rhsSpanBuf.span().data() == nullptr);
    assert(rhsSpanBuf.span().size() == 0);
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
