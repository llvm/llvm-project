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
//     : public basic_streambuf<charT, traits> {

//   Test stream operations inherited from `basic_istream` and `basic_ostream`

#include <cassert>
#include <concepts>
#include <span>
#include <spanstream>
#include <string>
#include <sstream>

#include "constexpr_char_traits.h"
#include "test_convertible.h"
#include "test_macros.h"

#include "../helper_macros.h"

#include <print>
#include <iostream>

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

  constexpr std::basic_string_view<CharT, TraitsT> sv{SV("zmt 94 hkt 82 pir ")};
  CharT arr[sv.size() + 1];
  initialize_array(arr, sv);
  // if constexpr (std::same_as<CharT, char>)
  //   strncpy(arr, sv.data(), sv.size() + 1);
  // else
  //   wcsncpy(arr, sv.data(), sv.size() + 1);

  std::span<CharT> sp{arr};

  // if constexpr (std::same_as<CharT, char>) {
  //   std::println(stderr, "{}", sp.data());
  //   std::println(stderr, "{}", sp);
  // } else {
  //   // std::println(stderr, "{}", sp.data());
  //   // std::println(stderr, "{}", sp);
  //   // std::println(stderr, L"{}", L"sv.data()");
  //   std::wcerr << std::format(L"L {}", sp.data()) << std::endl;
  //   std::wcerr << std::format(L"L {}", sp) << std::endl;
  // }

  // Mode: default
  {
    SpStream spSt(sp);
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

    if constexpr (std::same_as<CharT, char>) {
      std::println(stderr, "str1 '{}'", str1);
      std::println(stderr, "str2 '{}'", str2);
      std::println(stderr, "str3 '{}'", str3);
    } else {
      // std::println(stderr, "{}", sp.data());
      // std::println(stderr, "{}", sp);
      // std::println(stderr, L"{}", L"sv.data()");
      std::wcerr << std::format(L"L str1 '{}'", str1) << std::endl;
      std::wcerr << std::format(L"L str2 '{}'", str2) << std::endl;
      std::wcerr << std::format(L"L str3 '{}'", str3) << std::endl;
    }
    assert(str1 == CS("zmt"));
    assert(i1 == 94);
    assert(str2 == CS("hkt"));
    assert(i2 == 82);
    assert(str3 == CS("pir"));
  }
  // std::cerr << "========================================" << std::endl;
  {
    std::basic_istringstream<CharT> spSt{sv.data()};
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

    if constexpr (std::same_as<CharT, char>) {
      std::println(stderr, "- str1 '{}'", str1);
      std::println(stderr, "str2 '{}'", str2);
      std::println(stderr, "str3 '{}'", str3);
    } else {
      // std::cerr << "lfasdfasdfasdfasd" << std::endl;
      // std::wcerr << std::format(L"L - str1 '{}'", L"+++++++++++++++++++++++++") << std::endl;
      std::wcerr << std::format(L"L - str1 '{}'", str1) << std::endl;
      std::wcerr << std::format(L"L str2 '{}'", str2) << std::endl;
      std::wcerr << std::format(L"L str3 '{}'", str3) << std::endl;
    }
  }

  //   // Mode: default
  //   {
  //     SpStream rhsSpSt{sp};
  //     SpStream spSt(std::move(rhsSpSt));
  //     assert(spSt.span().data() == arr);
  //     assert(spSt.span().empty());
  //     assert(spSt.span().size() == 0);
  //   }
  //   // Mode: `ios_base::in`
  //   {
  //     SpStream rhsSpSt{sp, std::ios_base::in};
  //     SpStream spSt(std::move(rhsSpSt));
  //     assert(spSt.span().data() == arr);
  //     assert(!spSt.span().empty());
  //     assert(spSt.span().size() == 4);
  //   }
  //   // Mode `ios_base::out`
  //   {
  //     SpStream rhsSpSt{sp, std::ios_base::out};
  //     SpStream spSt(std::move(rhsSpSt));
  //     assert(spSt.span().data() == arr);
  //     assert(spSt.span().empty());
  //     assert(spSt.span().size() == 0);
  //   }
  //   // Mode: multiple
  //   {
  //     SpStream rhsSpSt{sp, std::ios_base::in | std::ios_base::out | std::ios_base::binary};
  //     SpStream spSt(std::move(rhsSpSt));
  //     assert(spSt.span().data() == arr);
  //     assert(spSt.span().empty());
  //     assert(spSt.span().size() == 0);
  //   }
}

int main(int, char**) {
  test<char>();
  // test<char, constexpr_char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
  // test<wchar_t, constexpr_char_traits<wchar_t>>();
#endif
  // std::println(stderr, "fasdfas");
  // std::println(std::cerr, "fasdfasdfasd{}", "-----");
  // std::println(std::cout, "fasdfasdfasd{}", "-----");
  // std::println(std::wcout, L"fasdfasdfasd{}", L"-----");
  // std::println(std::wcerr, L"fasdfasdfasd{}", L"-----");

  // assert(false);

  return 0;
}
