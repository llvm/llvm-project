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
//     pos_type seekpos(pos_type sp,
//                      ios_base::openmode which = ios_base::in | ios_base::out) override;

#include <cassert>
#include <span>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_macros.h"

// `seekpos` is the same as `seekoff` with `ios_base::beg`
template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpBuf = std::basic_spanbuf<CharT, TraitsT>;

  constexpr auto no_mode = 0;

  // Empty `span`
  {
    std::span<CharT> sp;

    // For an empty span:
    //    0 is an in-range offset value for an empty span
    //    3 is an out-of-range offset value

    // Mode: default (`in` | `out`)
    {
      SpBuf spBuf{sp};

      assert(spBuf.pubseekpos(0, std::ios_base::in) == 0);
      assert(spBuf.pubseekpos(3, std::ios_base::in) == -1);

      assert(spBuf.pubseekpos(0, std::ios_base::out) == 0);
      assert(spBuf.pubseekpos(3, std::ios_base::out) == -1);

      // Default parameter value `openmode`
      assert(spBuf.pubseekpos(0) == 0);
      assert(spBuf.pubseekpos(3) == -1);

      // No mode
      assert(spBuf.pubseekpos(0, no_mode) == 0);
      assert(spBuf.pubseekpos(3, no_mode) == -1);
    }
    // Mode: `in`
    {
      SpBuf spBuf{sp, std::ios_base::in};

      assert(spBuf.pubseekpos(0, std::ios_base::in) == 0);
      assert(spBuf.pubseekpos(3, std::ios_base::in) == -1);

      assert(spBuf.pubseekpos(0, std::ios_base::out) == 0);
      assert(spBuf.pubseekpos(3, std::ios_base::out) == -1);

      // Default parameter value `openmode`
      assert(spBuf.pubseekpos(0) == 0);
      assert(spBuf.pubseekpos(3) == -1);

      // No mode
      assert(spBuf.pubseekpos(0, no_mode) == 0);
      assert(spBuf.pubseekpos(3, no_mode) == -1);
    }
    // Mode: `out`
    {
      SpBuf spBuf{sp, std::ios_base::out};

      assert(spBuf.pubseekpos(0, std::ios_base::in) == 0);
      assert(spBuf.pubseekpos(3, std::ios_base::in) == -1);

      assert(spBuf.pubseekpos(0, std::ios_base::out) == 0);
      assert(spBuf.pubseekpos(3, std::ios_base::out) == -1);

      // Default parameter value `openmode`
      assert(spBuf.pubseekpos(0) == 0);
      assert(spBuf.pubseekpos(3) == -1);

      // No mode
      assert(spBuf.pubseekpos(0, no_mode) == 0);
      assert(spBuf.pubseekpos(3, no_mode) == -1);
    }
  }

  // Non-empty `span`
  {
    CharT arr[10];
    std::span sp{arr};

    // For an empty span:
    //    999 is an out-of-range offset value

    // Mode: default (`in` | `out`)
    {
      SpBuf spBuf{sp};

      assert(spBuf.pubseekpos(999, std::ios_base::in) == -1);

      assert(spBuf.pubseekpos(999, std::ios_base::out) == -1);

      assert(spBuf.pubseekpos(0, std::ios_base::in) == 0);
      assert(spBuf.pubseekpos(3, std::ios_base::in) == 3);

      assert(spBuf.pubseekpos(0, std::ios_base::out) == 0);
      assert(spBuf.pubseekpos(3, std::ios_base::out) == 3);

      // Default parameter value `openmode`
      assert(spBuf.pubseekpos(999) == -1);

      assert(spBuf.pubseekpos(0) == 0);
      assert(spBuf.pubseekpos(3) == 3);

      // No mode
      assert(spBuf.pubseekpos(-999, no_mode) == -1);
      assert(spBuf.pubseekpos(999, no_mode) == -1);

      assert(spBuf.pubseekpos(0, no_mode) == 0);
      assert(spBuf.pubseekpos(3, no_mode) == 3);
    }
    // Mode: `in`
    {
      SpBuf spBuf{sp, std::ios_base::in};

      assert(spBuf.pubseekpos(999, std::ios_base::in) == -1);

      assert(spBuf.pubseekpos(999, std::ios_base::out) == -1);

      assert(spBuf.pubseekpos(0, std::ios_base::in) == 0);
      assert(spBuf.pubseekpos(3, std::ios_base::in) == 3);

      assert(spBuf.pubseekpos(0, std::ios_base::out) == 0);
      assert(spBuf.pubseekpos(3, std::ios_base::out) == -1);

      // Default parameter value `openmode`
      assert(spBuf.pubseekpos(999) == -1);

      assert(spBuf.pubseekpos(0) == 0);
      assert(spBuf.pubseekpos(3) == -1);

      // No mode
      assert(spBuf.pubseekpos(999, no_mode) == -1);

      assert(spBuf.pubseekpos(0, no_mode) == 0);
      assert(spBuf.pubseekpos(3, no_mode) == 3);
    }
    // Mode: `out`
    {
      SpBuf spBuf{sp, std::ios_base::out};

      assert(spBuf.pubseekpos(999, std::ios_base::in) == -1);
      
      assert(spBuf.pubseekpos(999, std::ios_base::out) == -1);

      assert(spBuf.pubseekpos(0, std::ios_base::in) == 0);
      assert(spBuf.pubseekpos(3, std::ios_base::in) == -1);

      assert(spBuf.pubseekpos(0, std::ios_base::out) == 0);
      assert(spBuf.pubseekpos(3, std::ios_base::out) == 3);

      // Default parameter value `openmode`
      assert(spBuf.pubseekpos(999) == -1);
      assert(spBuf.pubseekpos(0) == 0);
      assert(spBuf.pubseekpos(3) == -1);

      // No mode
      assert(spBuf.pubseekpos(999, no_mode) == -1);
      assert(spBuf.pubseekpos(0, no_mode) == 0);
      assert(spBuf.pubseekpos(3, no_mode) == 3);
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
