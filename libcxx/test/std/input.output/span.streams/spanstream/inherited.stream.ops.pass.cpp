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

    assert(spSt);
    assert(!spSt.bad());
    assert(!spSt.fail());
    assert(spSt.good());
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

    assert(spSt);
    assert(!spSt.bad());
    assert(!spSt.fail());
    assert(spSt.good());

    // Write to stream
    constexpr std::basic_string_view<CharT, TraitsT> sv1{SV("year 2024")};
    spSt << sv1;

    assert(spSt.span().size() == sv1.size());
    assert(spSt);
    assert(!spSt.bad());
    assert(!spSt.fail());
    assert(spSt.good());

    // Read from stream
    spSt.seekg(0);
    std::basic_string<CharT, TraitsT> str4;
    spSt >> str4;
    int i4;
    spSt >> i4;

    assert(str4 == CS("year"));
    assert(i4 == 2024);

    spSt >> i4;

    assert(!spSt);
    assert(!spSt.bad());
    assert(spSt.fail());
    assert(!spSt.good());

    spSt.clear();

    assert(spSt);
    assert(!spSt.bad());
    assert(!spSt.fail());
    assert(spSt.good());

    // Write to stream
    spSt << CS("94");
    spSt << 84;

    assert(spSt);
    assert(!spSt.bad());
    assert(!spSt.fail());
    assert(spSt.good());

    assert(spSt.span().size() == sv1.size() + 4);
    std::basic_string<CharT, TraitsT> expectedStr1{spSt.span().data(), std::size_t{spSt.span().size()}};
    assert(expectedStr1 == CS("year 20249484"));

    // Write to stream with overflow
    constexpr std::basic_string_view<CharT, TraitsT> sv2{
        SV("This string should overflow! This string should overflow!")};
    spSt << sv2;
    assert(spSt.span().size() == arrSize);
    std::basic_string<CharT, TraitsT> expectedStr2{spSt.span().data(), std::size_t{spSt.span().size()}};
    assert(expectedStr2 == CS("year 20249484This string shoul"));

    assert(!spSt);
    assert(spSt.bad());
    assert(spSt.fail());
    assert(!spSt.good());
  }
#if 0
  // Mode: `in`
  {
    SpStream spSt{sp, std::ios_base::in};
    assert(spSt);
    assert(spSt.good());
    assert(spSt.span().size() == arrSize);

     std::basic_string<CharT, TraitsT> expectedStr0{spSt.span().data(), std::size_t{spSt.span().size()}};
     std::cout << expectedStr0 << std::endl;

    // // Read from stream
    // std::basic_string<CharT, TraitsT> str1;
    // spSt >> str1;
    // int i1;
    // spSt >> i1;
    // std::basic_string<CharT, TraitsT> str2;
    // spSt >> str2;
    // int i2;
    // spSt >> i2;
    // std::basic_string<CharT, TraitsT> str3;
    // spSt >> str3;
    // int i3;
    // spSt >> i3;

    // assert(spSt.good());
    // assert(str1 == CS("zmt"));
    // assert(i1 == 94);
    // assert(str2 == CS("hkt"));
    // assert(i2 == 82);
    // assert(str3 == CS("pir"));
    // assert(i3 == 43);

    // Write to stream
    constexpr std::basic_string_view<CharT, TraitsT> sv1{SV("year 2024")};
    spSt << sv1;

    std::cout << spSt.span().size() << std::endl;
    assert(spSt.span().size() == sv1.size());
    assert(spSt.good());

    // Read from stream
    spSt.seekg(0);
    std::basic_string<CharT, TraitsT> str4;
    spSt >> str4;
    int i4;
    spSt >> i4;

    assert(str4 == CS("year"));
    assert(i4 == 2024);

    spSt >> i4;
    assert(spSt);
    assert(!spSt.bad());
    assert(spSt.fail());
    assert(!spSt.good());

     spSt.clear();
    assert(spSt);
    assert(!spSt.bad());
    assert(!spSt.fail());
    assert(spSt.good());
    // Write to stream
    spSt << CS("94");
    spSt << 84;

    assert(spSt);
    assert(!spSt.bad());
    assert(!spSt.fail());
    assert(spSt.good());    assert(spSt.span().size() == sv1.size() + 4);
    std::basic_string<CharT, TraitsT> expectedStr1{spSt.span().data(), std::size_t{spSt.span().size()}};
    assert(expectedStr1 == CS("year 20249484"));

    // Write to stream with overflow
    constexpr std::basic_string_view<CharT, TraitsT> sv2{
        SV("This string should overflow! This string should overflow!")};
    spSt << sv2;
    assert(spSt.span().size() == arrSize);
    std::basic_string<CharT, TraitsT> expectedStr2{spSt.span().data(), std::size_t{spSt.span().size()}};
    assert(expectedStr2 == CS("year 20249484This string shoul"));
    assert(spSt);
    assert(!spSt.bad());
    assert(spSt.fail());
    assert(!spSt.good());  }
  // Mode `out`
  {
    SpStream spSt{sp, std::ios_base::out};
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

    assert(spSt.good());
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
    assert(spSt.good());

    // Read from stream
    spSt.seekg(0);
    std::basic_string<CharT, TraitsT> str4;
    spSt >> str4;
    int i4;
    spSt >> i4;

    assert(str4 == CS("year"));
    assert(i4 == 2024);

    spSt >> i4;
    assert(spSt.fail());
    spSt.clear();
    assert(spSt.good());

    // Write to stream
    spSt << CS("94");
    spSt << 84;

    assert(spSt.good());
    assert(spSt.span().size() == sv1.size() + 4);
    std::basic_string<CharT, TraitsT> expectedStr1{spSt.span().data(), std::size_t{spSt.span().size()}};
    assert(expectedStr1 == CS("year 20249484"));

    // Write to stream with overflow
    constexpr std::basic_string_view<CharT, TraitsT> sv2{
        SV("This string should overflow! This string should overflow!")};
    spSt << sv2;
    assert(spSt.span().size() == arrSize);
    std::basic_string<CharT, TraitsT> expectedStr2{spSt.span().data(), std::size_t{spSt.span().size()}};
    assert(expectedStr2 == CS("year 20249484This string shoul"));
    assert(spSt.fail());
  }
#endif
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
