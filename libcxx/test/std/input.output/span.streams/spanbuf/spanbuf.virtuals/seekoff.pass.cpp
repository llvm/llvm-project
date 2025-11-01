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
//
//     pos_type seekoff(off_type off, ios_base::seekdir way,
//                      ios_base::openmode which = ios_base::in | ios_base::out) override;

#include <cassert>
#include <span>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_macros.h"

template <typename CharT, typename TraitsT>
void test() {
  using SpanBuf = std::basic_spanbuf<CharT, TraitsT>;

  constexpr auto no_mode = 0;

  // Empty `span`
  {
    std::span<CharT> sp;

    // For an empty span:
    //   -3 is an out-of-range offset value
    //    0 is an in-range offset value for an empty span
    //    3 is an out-of-range offset value

    // Mode: default (`in` | `out`)
    {
      SpanBuf spanBuf{sp};

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::in) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::in) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, std::ios_base::in) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == -1);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::out) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::out) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, std::ios_base::out) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == -1);

      // Default parameter value `openmode`
      assert(spanBuf.pubseekoff(0, std::ios_base::beg) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur) == -1);
      assert(spanBuf.pubseekoff(0, std::ios_base::end) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end) == -1);

      // No mode
      assert(spanBuf.pubseekoff(0, std::ios_base::beg, no_mode) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, no_mode) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, no_mode) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(-1, std::ios_base::end, no_mode) == -1);
    }
    // Mode: `in`
    {
      SpanBuf spanBuf{sp, std::ios_base::in};

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::in) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::in) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, std::ios_base::in) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == -1);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::out) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::out) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, std::ios_base::out) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == -1);

      // Default parameter value `openmode`
      assert(spanBuf.pubseekoff(0, std::ios_base::beg) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur) == -1); // ???
      assert(spanBuf.pubseekoff(0, std::ios_base::end) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end) == -1);

      // No mode
      assert(spanBuf.pubseekoff(0, std::ios_base::beg, no_mode) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, no_mode) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, no_mode) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, no_mode) == -1);
    }
    // Mode: `out`
    {
      SpanBuf spanBuf{sp, std::ios_base::out};

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::in) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::in) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, std::ios_base::in) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == -1);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::out) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::out) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, std::ios_base::out) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == -1);

      // Default parameter value `openmode`
      assert(spanBuf.pubseekoff(0, std::ios_base::beg) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur) == -1); // ???
      assert(spanBuf.pubseekoff(0, std::ios_base::end) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end) == -1);

      // No mode
      assert(spanBuf.pubseekoff(0, std::ios_base::beg, no_mode) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, no_mode) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, no_mode) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, no_mode) == -1);
    }
  }

  // Non-empty `span`
  {
    CharT arr[10];
    std::span sp{arr};

    // For an empty span:
    //   -999 is an out-of-range offset value
    //    999 is an out-of-range offset value

    // Mode: default (`in` | `out`)
    {
      SpanBuf spanBuf{sp};

      assert(spanBuf.pubseekoff(-999, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::end, std::ios_base::in) == -1);

      assert(spanBuf.pubseekoff(999, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::end, std::ios_base::in) == -1);

      assert(spanBuf.pubseekoff(-999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::end, std::ios_base::out) == -1);

      assert(spanBuf.pubseekoff(999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::end, std::ios_base::out) == -1);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::in) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::in) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, std::ios_base::in) == 10); // ???

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == 3);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == 6);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == 7);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::out) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::out) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, std::ios_base::out) == 10); // ???

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == 3);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == 6);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == 7);

      // Default parameter value `openmode`
      assert(spanBuf.pubseekoff(-999, std::ios_base::beg) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::cur) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::end) == -1);

      assert(spanBuf.pubseekoff(999, std::ios_base::beg) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::cur) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::end) == -1);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur) == -1); // ???
      assert(spanBuf.pubseekoff(0, std::ios_base::end) == 10);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg) == 3);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end) == 7);

      // No mode
      assert(spanBuf.pubseekoff(-999, std::ios_base::beg, no_mode) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::end, no_mode) == -1);

      assert(spanBuf.pubseekoff(999, std::ios_base::beg, no_mode) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::end, no_mode) == -1);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, no_mode) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, no_mode) == 10); // ???

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, no_mode) == 3);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, no_mode) == 7);
    }
    // Mode: `in`
    {
      SpanBuf spanBuf{sp, std::ios_base::in};

      assert(spanBuf.pubseekoff(-999, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::end, std::ios_base::in) == -1);

      assert(spanBuf.pubseekoff(999, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::end, std::ios_base::in) == -1);

      assert(spanBuf.pubseekoff(-999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::end, std::ios_base::out) == -1);

      assert(spanBuf.pubseekoff(999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::end, std::ios_base::out) == -1);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::in) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::in) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, std::ios_base::in) == 10); // ???

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == 3);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == 6);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == 7);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::out) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::out) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, std::ios_base::out) == -1); // ???

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == -1);

      // Default parameter value `openmode`
      assert(spanBuf.pubseekoff(-999, std::ios_base::beg) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::cur) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::end) == -1);

      assert(spanBuf.pubseekoff(999, std::ios_base::beg) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::cur) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::end) == -1);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur) == -1); // ???
      assert(spanBuf.pubseekoff(0, std::ios_base::end) == -1); // ???

      assert(spanBuf.pubseekoff(3, std::ios_base::beg) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end) == -1);

      // No mode
      assert(spanBuf.pubseekoff(-999, std::ios_base::beg, no_mode) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::end, no_mode) == -1);

      assert(spanBuf.pubseekoff(999, std::ios_base::beg, no_mode) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::end, no_mode) == -1);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, no_mode) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, no_mode) == 10); // ???

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, no_mode) == 3);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, no_mode) == 7);
    }
    // Mode: `out`
    {
      SpanBuf spanBuf{sp, std::ios_base::out};

      assert(spanBuf.pubseekoff(-999, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::end, std::ios_base::in) == -1);

      assert(spanBuf.pubseekoff(999, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::end, std::ios_base::in) == -1);

      assert(spanBuf.pubseekoff(-999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::end, std::ios_base::out) == -1);

      assert(spanBuf.pubseekoff(999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::end, std::ios_base::out) == -1);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::in) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::in) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, std::ios_base::in) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == -1);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::out) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::out) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, std::ios_base::out) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == 3);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == 6);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == 3);

      // Default parameter value `openmode`
      assert(spanBuf.pubseekoff(-999, std::ios_base::beg) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::cur) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::end) == -1);

      assert(spanBuf.pubseekoff(999, std::ios_base::beg) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::cur) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::end) == -1);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur) == -1); // ???
      assert(spanBuf.pubseekoff(0, std::ios_base::end) == 0);

      assert(spanBuf.pubseekoff(3, std::ios_base::beg) == -1);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end) == -1);

      // No mode
      assert(spanBuf.pubseekoff(-999, std::ios_base::beg, no_mode) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(-999, std::ios_base::end, no_mode) == -1);

      assert(spanBuf.pubseekoff(999, std::ios_base::beg, no_mode) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(999, std::ios_base::end, no_mode) == -1);

      assert(spanBuf.pubseekoff(0, std::ios_base::beg, no_mode) == 0);
      assert(spanBuf.pubseekoff(0, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(0, std::ios_base::end, no_mode) == 0); // ???

      assert(spanBuf.pubseekoff(3, std::ios_base::beg, no_mode) == 3);
      assert(spanBuf.pubseekoff(3, std::ios_base::cur, no_mode) == -1);
      assert(spanBuf.pubseekoff(-3, std::ios_base::end, no_mode) == -1); // ???
    }
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
