//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format

// <format>

// Class typedefs:
// template<class Out, class charT>
// class basic_format_context {
// public:
//   using iterator = Out
//   using char_type = charT;
//   template<class T> using formatter_type = formatter<T, charT>;
// }
//
// Namespace std typedefs:
// using format_context = basic_format_context<unspecified, char>;
// using wformat_context = basic_format_context<unspecified, wchar_t>;

#include <format>
#include <string>
#include <string_view>
#include <type_traits>

#include "test_macros.h"

template <class OutIt, class CharT>
constexpr void test() {
  static_assert(
      std::is_same_v<typename std::basic_format_context<OutIt, CharT>::iterator,
                     OutIt>);
  static_assert(
      std::is_same_v<
          typename std::basic_format_context<OutIt, CharT>::char_type, CharT>);
  static_assert(std::is_same_v<typename std::basic_format_context<
                                   OutIt, CharT>::template formatter_type<bool>,
                               std::formatter<bool, CharT>>);
  static_assert(
      std::is_same_v<typename std::basic_format_context<
                         OutIt, CharT>::template formatter_type<CharT>,
                     std::formatter<CharT, CharT>>);
  static_assert(std::is_same_v<typename std::basic_format_context<
                                   OutIt, CharT>::template formatter_type<int>,
                               std::formatter<int, CharT>>);
  static_assert(
      std::is_same_v<typename std::basic_format_context<
                         OutIt, CharT>::template formatter_type<unsigned>,
                     std::formatter<unsigned, CharT>>);
  static_assert(
      std::is_same_v<typename std::basic_format_context<
                         OutIt, CharT>::template formatter_type<long long>,
                     std::formatter<long long, CharT>>);
  static_assert(
      std::is_same_v<typename std::basic_format_context<OutIt, CharT>::
                         template formatter_type<unsigned long long>,
                     std::formatter<unsigned long long, CharT>>);
#ifndef TEST_HAS_NO_INT128
  static_assert(
      std::is_same_v<typename std::basic_format_context<
                         OutIt, CharT>::template formatter_type<__int128_t>,
                     std::formatter<__int128_t, CharT>>);
  static_assert(
      std::is_same_v<typename std::basic_format_context<
                         OutIt, CharT>::template formatter_type<__uint128_t>,
                     std::formatter<__uint128_t, CharT>>);
#endif
  static_assert(
      std::is_same_v<typename std::basic_format_context<
                         OutIt, CharT>::template formatter_type<float>,
                     std::formatter<float, CharT>>);
  static_assert(
      std::is_same_v<typename std::basic_format_context<
                         OutIt, CharT>::template formatter_type<double>,
                     std::formatter<double, CharT>>);
  static_assert(
      std::is_same_v<typename std::basic_format_context<
                         OutIt, CharT>::template formatter_type<long double>,
                     std::formatter<long double, CharT>>);
  static_assert(
      std::is_same_v<typename std::basic_format_context<
                         OutIt, CharT>::template formatter_type<const CharT*>,
                     std::formatter<const CharT*, CharT>>);
  static_assert(
      std::is_same_v<typename std::basic_format_context<OutIt, CharT>::
                         template formatter_type<std::basic_string_view<CharT>>,
                     std::formatter<std::basic_string_view<CharT>, CharT>>);
  static_assert(
      std::is_same_v<typename std::basic_format_context<
                         OutIt, CharT>::template formatter_type<const void*>,
                     std::formatter<const void*, CharT>>);
}

constexpr void test() {
  test<std::back_insert_iterator<std::basic_string<char>>, char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<std::back_insert_iterator<std::basic_string<wchar_t>>, wchar_t>();
#endif
  test<std::back_insert_iterator<std::basic_string<char8_t>>, char8_t>();
  test<std::back_insert_iterator<std::basic_string<char16_t>>, char16_t>();
  test<std::back_insert_iterator<std::basic_string<char32_t>>, char32_t>();
}

template <class, class>
constexpr bool is_basic_format_context_specialization = false;
template <class It, class CharT>
constexpr bool is_basic_format_context_specialization<std::basic_format_context<It, CharT>, CharT> = true;

static_assert(is_basic_format_context_specialization<std::format_context, char>);
LIBCPP_STATIC_ASSERT(
    std::is_same_v<
        std::format_context,
        std::basic_format_context<
            std::back_insert_iterator<std::basic_string<char>>, char>>);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(is_basic_format_context_specialization<std::wformat_context, wchar_t>);
LIBCPP_STATIC_ASSERT(
    std::is_same_v<
        std::wformat_context,
        std::basic_format_context<
            std::back_insert_iterator<std::basic_string<wchar_t>>, wchar_t>>);
#endif

// Required for MSVC internal test runner compatibility.
int main(int, char**) { return 0; }
