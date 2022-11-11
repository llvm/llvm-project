//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-has-no-incomplete-format

// <format>

// template<class T, class charT>
// concept formattable = ...

#include <array>
#include <bitset>
#include <bitset>
#include <chrono>
#include <complex>
#include <concepts>
#include <deque>
#include <format>
#include <forward_list>
#include <list>
#include <memory>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <stack>
#include <span>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <valarray>
#include <variant>

#include "test_macros.h"

#ifndef TEST_HAS_NO_FILESYSTEM_LIBRARY
#  include <filesystem>
#endif
#ifndef TEST_HAS_NO_LOCALIZATION
#  include <regex>
#endif
#ifndef TEST_HAS_NO_THREADS
#  include <thread>
#endif

template <class T, class CharT>
void assert_is_not_formattable() {
  static_assert(!std::formattable<T, CharT>);
}

template <class T, class CharT>
void assert_is_formattable() {
  // Only formatters for CharT == char || CharT == wchar_t are enabled for the
  // standard formatters. When CharT is a different type the formatter should
  // be disabled.
  if constexpr (std::same_as<CharT, char>
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
                || std::same_as<CharT, wchar_t>
#endif
  )
    static_assert(std::formattable<T, CharT>);
  else
    assert_is_not_formattable<T, CharT>();
}

// Tests for P0645 Text Formatting
template <class CharT>
void test_P0645() {
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // Tests the special formatter that converts a char to a wchar_t.
  assert_is_formattable<char, wchar_t>();
#endif
  assert_is_formattable<CharT, CharT>();

  assert_is_formattable<CharT*, CharT>();
  assert_is_formattable<const CharT*, CharT>();
  assert_is_formattable<CharT[42], CharT>();
  assert_is_formattable<std::basic_string<CharT>, CharT>();
  assert_is_formattable<std::basic_string_view<CharT>, CharT>();

  assert_is_formattable<bool, CharT>();

  assert_is_formattable<signed char, CharT>();
  assert_is_formattable<signed short, CharT>();
  assert_is_formattable<signed int, CharT>();
  assert_is_formattable<signed long, CharT>();
  assert_is_formattable<signed long long, CharT>();
#ifndef TEST_HAS_NO_INT128
  assert_is_formattable<__int128_t, CharT>();
#endif

  assert_is_formattable<unsigned char, CharT>();
  assert_is_formattable<unsigned short, CharT>();
  assert_is_formattable<unsigned int, CharT>();
  assert_is_formattable<unsigned long, CharT>();
  assert_is_formattable<unsigned long long, CharT>();
#ifndef TEST_HAS_NO_INT128
  assert_is_formattable<__uint128_t, CharT>();
#endif

  assert_is_formattable<float, CharT>();
  assert_is_formattable<double, CharT>();
  assert_is_formattable<long double, CharT>();

  assert_is_formattable<std::nullptr_t, CharT>();
  assert_is_formattable<void*, CharT>();
  assert_is_formattable<const void*, CharT>();
}

// Tests for P1361 Integration of chrono with text formatting
//
// Some tests are commented out since these types haven't been implemented in
// chrono yet. After P1361 has been implemented these formatters should be all
// enabled.
template <class CharT>
void test_P1361() {
// The chrono formatters require localization support.
// [time.format]/7
//   If the chrono-specs is omitted, the chrono object is formatted as if by
//   streaming it to std::ostringstream os with the formatting
//   locale imbued and copying os.str() through the output iterator of the
//   context with additional padding and adjustments as specified by the format
//   specifiers.
// In libc++ std:::ostringstream requires localization support.
#ifndef TEST_HAS_NO_LOCALIZATION

  assert_is_formattable<std::chrono::microseconds, CharT>();

  assert_is_not_formattable<std::chrono::sys_time<std::chrono::microseconds>, CharT>();
  //assert_is_formattable<std::chrono::utc_time<std::chrono::microseconds>, CharT>();
  //assert_is_formattable<std::chrono::tai_time<std::chrono::microseconds>, CharT>();
  //assert_is_formattable<std::chrono::gps_time<std::chrono::microseconds>, CharT>();
  assert_is_not_formattable<std::chrono::file_time<std::chrono::microseconds>, CharT>();
  assert_is_not_formattable<std::chrono::local_time<std::chrono::microseconds>, CharT>();

  assert_is_formattable<std::chrono::day, CharT>();
  assert_is_formattable<std::chrono::month, CharT>();
  assert_is_formattable<std::chrono::year, CharT>();

  assert_is_formattable<std::chrono::weekday, CharT>();
  assert_is_not_formattable<std::chrono::weekday_indexed, CharT>();
  assert_is_not_formattable<std::chrono::weekday_last, CharT>();

  assert_is_not_formattable<std::chrono::month_day, CharT>();
  assert_is_not_formattable<std::chrono::month_day_last, CharT>();
  assert_is_not_formattable<std::chrono::month_weekday, CharT>();
  assert_is_not_formattable<std::chrono::month_weekday_last, CharT>();

  assert_is_not_formattable<std::chrono::year_month, CharT>();
  assert_is_not_formattable<std::chrono::year_month_day, CharT>();
  assert_is_not_formattable<std::chrono::year_month_day_last, CharT>();
  assert_is_not_formattable<std::chrono::year_month_weekday, CharT>();
  assert_is_not_formattable<std::chrono::year_month_weekday_last, CharT>();

  assert_is_not_formattable<std::chrono::hh_mm_ss<std::chrono::microseconds>, CharT>();

  //assert_is_formattable<std::chrono::sys_info, CharT>();
  //assert_is_formattable<std::chrono::local_info, CharT>();

  //assert_is_formattable<std::chrono::zoned_time, CharT>();

#endif // TEST_HAS_NO_LOCALIZATION
}

// Tests for P1636 Formatters for library types
//
// The paper hasn't been voted in so currently all formatters are disabled.
// TODO validate whether the test is correct after the paper has been accepted.
template <class CharT>
void test_P1636() {
  assert_is_not_formattable<std::basic_streambuf<CharT>, CharT>();
  assert_is_not_formattable<std::bitset<42>, CharT>();
  assert_is_not_formattable<std::complex<double>, CharT>();
  assert_is_not_formattable<std::error_code, CharT>();
#ifndef TEST_HAS_NO_FILESYSTEM_LIBRARY
  assert_is_not_formattable<std::filesystem::path, CharT>();
#endif
  assert_is_not_formattable<std::shared_ptr<int>, CharT>();
#ifndef TEST_HAS_NO_LOCALIZATION
  assert_is_not_formattable<std::sub_match<CharT*>, CharT>();
#endif
#ifndef TEST_HAS_NO_THREADS
  assert_is_not_formattable<std::thread::id, CharT>();
#endif
  assert_is_not_formattable<std::unique_ptr<int>, CharT>();
}

// Tests for P2286 Formatting ranges
//
// The paper hasn't been voted in so currently all formatters are disabled.
// TODO validate whether the test is correct after the paper has been accepted.
template <class CharT>
void test_P2286() {
  assert_is_not_formattable<std::array<int, 42>, CharT>();
  assert_is_not_formattable<std::vector<int>, CharT>();
  assert_is_not_formattable<std::deque<int>, CharT>();
  assert_is_not_formattable<std::forward_list<int>, CharT>();
  assert_is_not_formattable<std::list<int>, CharT>();

  assert_is_not_formattable<std::set<int>, CharT>();
  assert_is_not_formattable<std::map<int, int>, CharT>();
  assert_is_not_formattable<std::multiset<int>, CharT>();
  assert_is_not_formattable<std::multimap<int, int>, CharT>();

  assert_is_not_formattable<std::unordered_set<int>, CharT>();
  assert_is_not_formattable<std::unordered_map<int, int>, CharT>();
  assert_is_not_formattable<std::unordered_multiset<int>, CharT>();
  assert_is_not_formattable<std::unordered_multimap<int, int>, CharT>();

  assert_is_not_formattable<std::stack<int>, CharT>();
  assert_is_not_formattable<std::queue<int>, CharT>();
  assert_is_not_formattable<std::priority_queue<int>, CharT>();

  assert_is_not_formattable<std::span<int>, CharT>();

  assert_is_not_formattable<std::valarray<int>, CharT>();

  assert_is_not_formattable<std::pair<int, int>, CharT>();
  assert_is_not_formattable<std::tuple<int>, CharT>();
}

class c {
  void f();
  void fc() const;
  static void sf();
};
enum e { a };
enum class ec { a };
template <class CharT>
void test_disabled() {
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  assert_is_not_formattable<const char*, wchar_t>();
#endif
  assert_is_not_formattable<const char*, char8_t>();
  assert_is_not_formattable<const char*, char16_t>();
  assert_is_not_formattable<const char*, char32_t>();

  assert_is_not_formattable<c, CharT>();
  assert_is_not_formattable<const c, CharT>();
  assert_is_not_formattable<volatile c, CharT>();
  assert_is_not_formattable<const volatile c, CharT>();

  assert_is_not_formattable<e, CharT>();
  assert_is_not_formattable<const e, CharT>();
  assert_is_not_formattable<volatile e, CharT>();
  assert_is_not_formattable<const volatile e, CharT>();

  assert_is_not_formattable<ec, CharT>();
  assert_is_not_formattable<const ec, CharT>();
  assert_is_not_formattable<volatile ec, CharT>();
  assert_is_not_formattable<const volatile ec, CharT>();

  assert_is_not_formattable<int*, CharT>();
  assert_is_not_formattable<const int*, CharT>();
  assert_is_not_formattable<volatile int*, CharT>();
  assert_is_not_formattable<const volatile int*, CharT>();

  assert_is_not_formattable<c*, CharT>();
  assert_is_not_formattable<const c*, CharT>();
  assert_is_not_formattable<volatile c*, CharT>();
  assert_is_not_formattable<const volatile c*, CharT>();

  assert_is_not_formattable<e*, CharT>();
  assert_is_not_formattable<const e*, CharT>();
  assert_is_not_formattable<volatile e*, CharT>();
  assert_is_not_formattable<const volatile e*, CharT>();

  assert_is_not_formattable<ec*, CharT>();
  assert_is_not_formattable<const ec*, CharT>();
  assert_is_not_formattable<volatile ec*, CharT>();
  assert_is_not_formattable<const volatile ec*, CharT>();

  assert_is_not_formattable<void (*)(), CharT>();
  assert_is_not_formattable<void (c::*)(), CharT>();
  assert_is_not_formattable<void (c::*)() const, CharT>();

  assert_is_not_formattable<std::optional<int>, CharT>();
  assert_is_not_formattable<std::variant<int>, CharT>();

  assert_is_not_formattable<std::shared_ptr<c>, CharT>();
  assert_is_not_formattable<std::unique_ptr<c>, CharT>();

  assert_is_not_formattable<std::array<c, 42>, CharT>();
  assert_is_not_formattable<std::vector<c>, CharT>();
  assert_is_not_formattable<std::deque<c>, CharT>();
  assert_is_not_formattable<std::forward_list<c>, CharT>();
  assert_is_not_formattable<std::list<c>, CharT>();

  assert_is_not_formattable<std::set<c>, CharT>();
  assert_is_not_formattable<std::map<c, int>, CharT>();
  assert_is_not_formattable<std::multiset<c>, CharT>();
  assert_is_not_formattable<std::multimap<c, int>, CharT>();

  assert_is_not_formattable<std::unordered_set<c>, CharT>();
  assert_is_not_formattable<std::unordered_map<c, int>, CharT>();
  assert_is_not_formattable<std::unordered_multiset<c>, CharT>();
  assert_is_not_formattable<std::unordered_multimap<c, int>, CharT>();

  assert_is_not_formattable<std::stack<c>, CharT>();
  assert_is_not_formattable<std::queue<c>, CharT>();
  assert_is_not_formattable<std::priority_queue<c>, CharT>();

  assert_is_not_formattable<std::span<c>, CharT>();

  assert_is_not_formattable<std::valarray<c>, CharT>();

  assert_is_not_formattable<std::pair<c, int>, CharT>();
  assert_is_not_formattable<std::tuple<c>, CharT>();

  assert_is_not_formattable<std::optional<c>, CharT>();
  assert_is_not_formattable<std::variant<c>, CharT>();
}

template <class CharT>
void test() {
  test_P0645<CharT>();
  test_P1361<CharT>();
  test_P1636<CharT>();
  test_P2286<CharT>();
  test_disabled<CharT>();
}

void test() {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  test<char8_t>();
  test<char16_t>();
  test<char32_t>();

  test<int>();
}
