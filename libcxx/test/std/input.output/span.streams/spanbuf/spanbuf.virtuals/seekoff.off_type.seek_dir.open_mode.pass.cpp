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

#include <print>

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpBuf       = std::basic_spanbuf<CharT, TraitsT>;
  using SpBuffError = typename SpBuf::pos_type;

  const SpBuffError error{-1};

  // Empty `span`
  {
    std::span<CharT> sp;

    // Mode: default (`in` | `out`)
    {
      SpBuf spBuf{sp};
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::out) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::out) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::end, std::ios_base::out) == 0);
    }
    // Mode: `in`
    {
      SpBuf spBuf{sp, std::ios_base::in};
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::in) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::in) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::end, std::ios_base::in) == 0);
    }
    // Mode: `out`
    {
      SpBuf spBuf{sp, std::ios_base::out};
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::out) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::out) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::end, std::ios_base::out) == 0);
    }
    // Mode: `multiple`
    {
      SpBuf spBuf{sp, std::ios_base::in | std::ios_base::out | std::ios_base::binary};
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::out) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::out) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::end, std::ios_base::out) == 0);
    }
    // Mode: `ate`
    {
      SpBuf spBuf{sp, std::ios_base::out | std::ios_base::ate};
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(0, std::ios_base::beg, std::ios_base::in) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::cur, std::ios_base::in) == 0);
      assert(spBuf.pubseekoff(0, std::ios_base::end, std::ios_base::in) == 0);
    }
  }

  // Non-empty `span`
  {
    // Initialize with ASCII codes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    CharT arr[]{
        CharT{48}, CharT{49}, CharT{50}, CharT{51}, CharT{52}, CharT{53}, CharT{54}, CharT{55}, CharT{56}, CharT{57}};
    std::span sp{arr};
    // std::print(stderr, "-=----- {}", sp);
    // assert(false);

    // Mode: default (`in` | `out`)
    {
      // TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp);
      // TestSpanBuf<CharT, TraitsT> spBuf = std::move(static_cast<SpBuf&>(rhsSpBuf));
      // assert(rhsSpBuf.span().data() == arr);
      // assert(spBuf.span().data() == arr);
      // spBuf.check_postconditions(rhsSpBuf);
    }
    // Mode: `in`
    {
      SpBuf spBuf(sp, std::ios_base::in);
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in | std::ios_base::out) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == 3);
      assert(spBuf.sgetc() == 51);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == 6);
      assert(spBuf.sgetc() == 54);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == 7);
      assert(spBuf.sgetc() == 55);
    }
    // Mode: `out`
    {
      CharT resultArr[]{
          CharT{48}, CharT{49}, CharT{50}, CharT{51}, CharT{52}, CharT{53}, CharT{54}, CharT{55}, CharT{56}, CharT{57}};

      SpBuf spBuf(sp, std::ios_base::out);
      // std::println(stderr, "fasdfasdfasdfasdfasd 0 {}", spBuf.span());
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out | std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out | std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out | std::ios_base::in) == -1);
      assert(spBuf.pubseekoff(3, std::ios_base::beg, std::ios_base::out) == 3);
      // #if 0
      assert(spBuf.sputc(CharT{90}) == 90);
      // assert(spBuf.str() == "012a456789");
      resultArr[3] = CharT{90};
      // assert(spBuf.span() == std::span<CharT>{resultArr});
      std::println(stderr, "fasdfasdfasdfasdfasd 1 {}", spBuf.span());
      std::println(stderr, "fasdfasdfasdfasdfasd {}", spBuf.span().data());
      std::println(stderr, "fasdfasdfasdfasdfasd {}", spBuf.span().size());
      std::println(stderr, "fasdfasdfasdfasdfasd 2 {}", std::span<CharT>{resultArr});
      // assert(std::ranges::equal(spBuf.span(), std::span<CharT>{resultArr}));
      // assert(spBuf.pubseekoff(3, std::ios_base::cur, std::ios_base::out) == 7);
      assert(spBuf.sputc(CharT{77}) == 77);
      std::println(stderr,"fasdfasdfasdfasdfasd 5 {}", spBuf.span());
      // assert(spBuf.str() == "012a456b89");
      resultArr[7] = CharT{77};
      // assert(spBuf.span() == std::span<CharT>{resultArr});
      // assert(std::ranges::equal(spBuf.span(), std::span<CharT>{resultArr}));
      // std::println(stderr, "ljjklj {}",  static_cast<int>(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out)));
      assert(spBuf.pubseekoff(-3, std::ios_base::end, std::ios_base::out) == 2);
      std::println(stderr, "fasdfasdfasdfasdfasd 6 {}", spBuf.span());
      assert(spBuf.sputc(CharT{84}) == 84);
#if 0
      // assert(spBuf.str() == "012a456c89");
      resultArr[6] = CharT{84};
      assert(spBuf.span() == std::span<CharT>{resultArr});
#endif
      assert(false);
    }
    // Mode: multiple
    {
      // std::span<CharT> sp{arr};
      // TestSpanBuf<CharT, TraitsT> rhsSpBuf(sp, std::ios_base::in | std::ios_base::out | std::ios_base::binary);
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
  // test<nasty_char, nasty_char_traits>();
#endif
  test<char>();
  test<char, constexpr_char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // test<wchar_t>();
  // test<wchar_t, constexpr_char_traits<wchar_t>>();
#endif

  return 0;
}
