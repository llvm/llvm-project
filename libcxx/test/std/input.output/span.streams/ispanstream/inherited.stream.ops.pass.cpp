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
//   class basic_spanstream
//     : public basic_iostream<charT, traits> {

//   Test stream operations inherited from `basic_istream`

#include <algorithm>
#include <cassert>
#include <span>
#include <spanstream>
#include <string_view>

#include "constexpr_char_traits.h"
#include "test_macros.h"

#include "../helper_macros.h"
#include "../helper_types.h"

template <typename CharT, typename TraitsT>
void test_ispanstream(std::basic_ispanstream<CharT, TraitsT>& spSt, std::size_t size) {
  assert(spSt);
  assert(!spSt.bad());
  assert(!spSt.fail());
  assert(spSt.good());
  assert(spSt.span().size() == size);

  // Read from stream
  std::basic_string<CharT, TraitsT> str1;
  spSt >> str1;
  int i1;
  spSt >> i1;
  std::basic_string<CharT, TraitsT> str2;
  spSt >> str2;
  int i2;
  spSt >> i2;
  std::basic_string<CharT, TraitsT> str3;
  spSt >> str3;
  int i3;
  spSt >> i3;

  assert(str1 == CS("zmt"));
  assert(i1 == 94);
  assert(str2 == CS("hkt"));
  assert(i2 == 82);
  assert(str3 == CS("pir"));
  assert(i3 == 43);

  assert(spSt);
  assert(!spSt.bad());
  assert(!spSt.fail());
  assert(spSt.good());

  spSt.clear();

  assert(spSt);
  assert(!spSt.bad());
  assert(!spSt.fail());
  assert(spSt.good());
}

template <typename CharT, typename TraitsT>
void test() {
  using SpStream = std::basic_ispanstream<CharT, TraitsT>;

  constexpr std::basic_string_view<CharT, TraitsT> sv{SV("zmt 94 hkt 82 pir 43vr")};
  assert(sv.size() < 30UZ);

  {
    // Create a std::span test value
    CharT arr[30UZ]{};
    std::ranges::copy(sv, arr);

    std::span<CharT> sp{arr};

    // std::span` + Mode: default (`in`)
    {
      SpStream spSt(sp);
      test_ispanstream(spSt, 30UZ);
    }
    // std::span` + Mode: explicit `in`
    {
      SpStream spSt(sp, std::ios_base::in);
      test_ispanstream(spSt, 30UZ);
    }
  }

  {
    // Create a "Read Only Sequence" test value
    CharT arr[30UZ]{};
    std::ranges::copy(sv, arr);

    ReadOnlySpan<CharT, 30UZ> ros{arr};
    assert(ros.size() == 30UZ);

    {
      SpStream spSt(ros);
      test_ispanstream(spSt, 30UZ);
    }
    {
      SpStream spSt(std::move(ros));
      test_ispanstream(spSt, 30UZ);
    }
  }
}

int main(int, char**) {
  test<char, std::char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // test<wchar_t, std::char_traits<wchar_t>>();
#endif

  return 0;
}
