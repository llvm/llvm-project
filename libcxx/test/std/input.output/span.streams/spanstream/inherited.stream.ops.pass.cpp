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
//   class basic_spanstream
//     : public basic_iostream<charT, traits> {

//   Test stream operations inherited from `basic_istream` and `basic_ostream`

#include <cassert>
#include <concepts>
#include <span>
#include <spanstream>
#include <string>

#include "test_macros.h"

#include "../helper_macros.h"

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void initialize_array(CharT* arr, std::basic_string_view<CharT, TraitsT> sv) {
  if constexpr (std::same_as<CharT, char>)
    strncpy(arr, sv.data(), sv.size() + 1);
  else
    wcsncpy(arr, sv.data(), sv.size() + 1);
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpStream = std::basic_spanstream<CharT, TraitsT>;

  constexpr auto arrSize{30UZ};

  constexpr std::basic_string_view<CharT, TraitsT> sv{SV("zmt 94 hkt 82 pir 43vr")};
  assert(sv.size() < arrSize);

  CharT arr[arrSize]{};
  initialize_array(arr, sv);

  std::span<CharT> sp{arr};

  // Mode: default (`in` | `out`)
  {
    SpStream spSt(sp);
    assert(spSt.span().size() == 0);

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

    // Write to stream
    constexpr std::basic_string_view<CharT, TraitsT> sv1{SV("year 2024")};
    spSt << sv1;
    assert(spSt.span().size() == sv1.size());

    // Read from stream
    spSt.seekg(0);
    std::basic_string<CharT, TraitsT> str4;
    spSt >> str4;
    int i4;
    spSt >> i4;

    assert(str4 == CS("year"));
    assert(i4 == 2024);

    // Write to stream
    spSt << CS("94");
    spSt << 84;
    std::cout << spSt.span().size() << std::endl;
    assert(spSt.span().size() == sv1.size() + 4);

    // Write to stream with overflow
    constexpr std::basic_string_view<CharT, TraitsT> sv2{
        SV("This string should overflow! This string should overflow!")};
    spSt << sv2;
    assert(spSt.span().size() == arrSize);
  }
  // Mode: `in`
  {
    SpStream spSt{sp, std::ios_base::in};
    //TODO
    (void)spSt;
  }
  // Mode `out`
  {
    SpStream spSt{sp, std::ios_base::out};
    //TODO
    (void)spSt;
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
