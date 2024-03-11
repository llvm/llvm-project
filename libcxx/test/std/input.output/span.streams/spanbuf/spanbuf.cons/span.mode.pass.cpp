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
//
//     explicit basic_spanbuf(std::span<charT> s,
//                            ios_base::openmode which = ios_base::in | ios_base::out);

#include <cassert>
#include <span>
#include <spanstream>
#include <utility>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_convertible.h"
#include "test_macros.h"

#include "../../helper_macros.h"
#include "../../helper_types.h"

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test_sfinae() {
  using SpBuf = std::basic_spanbuf<CharT, TraitsT>;

  // Mode
  static_assert(std::constructible_from<SpBuf, const std::span<CharT>, std::ios_base::openmode>);
  static_assert(!test_convertible<SpBuf, const std::span<CharT>, std::ios_base::openmode>());

  // Non-mode
  static_assert(!std::constructible_from<SpBuf, const std::span<CharT>, const NonMode>);
  static_assert(!test_convertible<SpBuf, const std::span<CharT>, const NonMode>());
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpBuf = std::basic_spanbuf<CharT, TraitsT>;

  // Empty `span`
  {
    std::span<CharT> sp{};

    // Mode: default (`in` | `out`)
    {
      SpBuf spBuf(sp);
      assert(spBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);
    }
    {
      SpBuf spBuf(std::as_const(sp));
      assert(spBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);
    }
    // Mode: `in`
    {
      SpBuf spBuf(sp, std::ios_base::in);
      assert(spBuf.span().data() == nullptr);
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);
    }
    {
      SpBuf spBuf(std::as_const(sp), std::ios_base::in);
      assert(spBuf.span().data() == nullptr);
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);
    }
    // Mode: `out`
    {
      SpBuf spBuf(sp, std::ios_base::out);
      assert(spBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);
    }
    {
      SpBuf spBuf(std::as_const(sp), std::ios_base::out);
      assert(spBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);
    }
    // Mode: multiple
    {
      SpBuf spBuf(sp, std::ios_base::in | std::ios_base::out | std::ios_base::binary);
      assert(spBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);
    }
    {
      SpBuf spBuf(std::as_const(sp), std::ios_base::in | std::ios_base::out | std::ios_base::binary);
      assert(spBuf.span().data() == nullptr);
      // Mode `out` counts read characters
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);
    }
    // Mode: `ate`
    {
      SpBuf spBuf(sp, std::ios_base::out | std::ios_base::ate);
      assert(spBuf.span().data() == nullptr);
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);
    }
    {
      SpBuf spBuf(std::as_const(sp), std::ios_base::out | std::ios_base::ate);
      assert(spBuf.span().data() == nullptr);
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);
    }
  }

  // Non-empty `span`
  {
    CharT arr[4];
    std::span<CharT> sp{arr};

    // Mode: default (`in` | `out`)
    {
      SpBuf spBuf(sp);
      assert(spBuf.span().data() == arr);
      // Mode `out` counts read characters
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);
    }
    {
      SpBuf spBuf(std::as_const(sp));
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
    {
      SpBuf spBuf(std::as_const(sp), std::ios_base::in);
      assert(spBuf.span().data() == arr);
      assert(!spBuf.span().empty());
      assert(spBuf.span().size() == 4);
    }
    // Mode `out`
    {
      SpBuf spBuf(sp, std::ios_base::out);
      assert(spBuf.span().data() == arr);
      // Mode `out` counts read characters
      assert(spBuf.span().empty());
      assert(spBuf.span().size() == 0);
    }
    {
      SpBuf spBuf(std::as_const(sp), std::ios_base::out);
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
    {
      SpBuf spBuf(std::as_const(sp), std::ios_base::in | std::ios_base::out | std::ios_base::binary);
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
    {
      SpBuf spBuf(std::as_const(sp), std::ios_base::out | std::ios_base::ate);
      assert(spBuf.span().data() == arr);
      assert(!spBuf.span().empty());
      assert(spBuf.span().size() == 4);
    }
  }
}

int main(int, char**) {
#ifndef TEST_HAS_NO_NASTY_STRING
  test_sfinae<nasty_char, nasty_char_traits>();
#endif
  test_sfinae<char>();
  test_sfinae<char, constexpr_char_traits<char>>();
#ifndef TEST_HAS_NO_NASTY_STRING
  test<nasty_char, nasty_char_traits>();
#endif
  test<char>();
  test<char, constexpr_char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_sfinae<wchar_t>();
  test_sfinae<wchar_t, constexpr_char_traits<wchar_t>>();
  test<wchar_t>();
  test<wchar_t, constexpr_char_traits<wchar_t>>();
#endif

  return 0;
}
