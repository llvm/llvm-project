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

//     // [spanbuf.virtuals], overridden virtual functions
//
//     pos_type seekoff(off_type off, ios_base::seekdir way,
//                      ios_base::openmode which = ios_base::in | ios_base::out) override;

#include <algorithm>
#include <cassert>
#include <span>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_macros.h"

#include <iostream>

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpBuf = std::basic_spanbuf<CharT, TraitsT>;

  // Empty `span`
  {
    std::span<CharT> sp;

    // Mode: default (`in` | `out`)
    {
      SpBuf spBuf{sp};

      // Out-of-range
      assert(spBuf.pubseekoff(-999, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::end, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::end, std::ios_base::out) == -1);

      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == -1);

      assert(spBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::out) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::out) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::end, std::ios_base::out) == 0);

      // Default `in` && `out`
      assert(spBuf.pubseekoff(3, std::ios_base::beg) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end) == -1);
    }
    // Mode: `in`
    {
      SpBuf spBuf{sp, std::ios_base::in};

      // Out-of-range
      assert(spBuf.pubseekoff(-999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::end, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::end, std::ios_base::out) == -1);

      assert(spBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::in) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::in) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::end, std::ios_base::in) == 0);

      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == -1);

      // Default `in` && `out`
      assert(spBuf.pubseekoff(3, std::ios_base::beg) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end) == -1);
    }
    // Mode: `out`
    {
      SpBuf spBuf{sp, std::ios_base::out};

      // Out-of-range
      assert(spBuf.pubseekoff(-999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::end, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::end, std::ios_base::out) == -1);

      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == -1);

      assert(spBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::out) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::out) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::end, std::ios_base::out) == 0);

      // Default `in` && `out`
      assert(spBuf.pubseekoff(3, std::ios_base::beg) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end) == -1);
    }
    // Mode: `multiple`
    {
      SpBuf spBuf{sp, std::ios_base::in | std::ios_base::out | std::ios_base::binary};

      // Out-of-range
      assert(spBuf.pubseekoff(-999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::end, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::end, std::ios_base::out) == -1);

      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == -1);

      assert(spBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::out) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::out) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::end, std::ios_base::out) == 0);

      // Default `in` && `out`
      assert(spBuf.pubseekoff(3, std::ios_base::beg) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end) == -1);
    }
    // Mode: `ate`
    {
      SpBuf spBuf{sp, std::ios_base::out | std::ios_base::ate};

      // Out-of-range
      assert(spBuf.pubseekoff(-999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::end, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::end, std::ios_base::out) == -1);

      assert(spBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::in) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::in) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::end, std::ios_base::in) == 0);

      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == -1);

      // Default `in` && `out`
      assert(spBuf.pubseekoff(3, std::ios_base::beg) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end) == -1);
    }
  }

  // Non-empty `span`
  {
    CharT arr[10];
    std::span sp{arr};

    // Mode: default (`in` | `out`)
    {
      SpBuf spBuf{sp};

      // Out-of-range
      assert(spBuf.pubseekoff(-999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::end, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::end, std::ios_base::out) == -1);

      std::cerr << spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) << std::endl;
      // assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == -1);

      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == 3);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == 6);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == 3);

      // Default `in` && `out`
      assert(spBuf.pubseekoff(3, std::ios_base::beg) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end) == -1);
    }
    // Mode: `in`
    {
      SpBuf spBuf{sp, std::ios_base::in};

      // Out-of-range
      assert(spBuf.pubseekoff(-999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::end, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::end, std::ios_base::out) == -1);

      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == 3);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == 6);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == 7);

      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == -1);

      // Default `in` && `out`
      assert(spBuf.pubseekoff(3, std::ios_base::beg) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end) == -1);
    }
    // Mode: `out`
    {
      SpBuf spBuf{sp, std::ios_base::out};

      // Out-of-range
      assert(spBuf.pubseekoff(-999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-999, std::ios_base::end, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(999, std::ios_base::end, std::ios_base::out) == -1);

      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == -1);

      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == 3);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == 6);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == 3);

      // Default `in` && `out`
      assert(spBuf.pubseekoff(3, std::ios_base::beg) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end) == -1);
    }
    // Mode: multiple
    {
      // std::span<CharT> sp{arr};
      // TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp | std::ios_base::binary);
      // TestSpanBuf<CharT, TraitsT> spBuf = std::move(static_cast<SpBuf&>(rhsSpBuf));
      // assert(rhsSpBuf.span().data() == arr);
      // assert(spBuf.span().data() == arr);
      // spBuf.check_postconditions(rhsSpBuf);
    }
    // Mode: `ate`
    {
      // std::span<CharT> sp{arr};
      // TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::out | std::ios_base::ate);
      // TestSpanBuf<CharT, TraitsT> spBuf = std::move(static_cast<SpBuf&>(rhsSpBuf));
      // assert(rhsSpBuf.span().data() == arr);
      // assert(spBuf.span().data() == arr);
      // spBuf.check_postconditions(rhsSpBuf);
    }
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
