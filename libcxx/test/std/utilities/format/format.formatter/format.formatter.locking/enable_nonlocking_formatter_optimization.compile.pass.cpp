//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <format>

// template<class T>
// constexpr bool enable_nonlocking_formatter_optimization = false;

// Remarks: Pursuant to [namespace.std], users may specialize
// enable_nonlocking_formatter_optimization for cv-unqualified program-defined
// types. Such specializations shall be usable in constant expressions
// ([expr.const]) and have type const bool.

// [format.formatter.spec]
// In addition, for each type T for which a formatter specialization is provided
// above, each of the headers provides the following specialization:
//
// template<>
// inline constexpr bool enable_nonlocking_formatter_optimization<T> = true;

#include <array>
#include <bitset>
#include <bitset>
#include <chrono>
#include <complex>
#include <concepts>
#include <deque>
#include <filesystem>
#include <format>
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <span>
#include <stack>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <valarray>
#include <variant>

#include "test_macros.h"
#include "min_allocator.h"

#ifndef TEST_HAS_NO_LOCALIZATION
#  include <regex>
#endif
#ifndef TEST_HAS_NO_THREADS
#  include <thread>
#endif

// Tests for P0645 Text Formatting
template <class CharT>
void test_P0645() {
  static_assert(std::enable_nonlocking_formatter_optimization<CharT>);

  static_assert(std::enable_nonlocking_formatter_optimization<CharT*>);
  static_assert(std::enable_nonlocking_formatter_optimization<const CharT*>);
  static_assert(std::enable_nonlocking_formatter_optimization<CharT[42]>);

  static_assert(std::enable_nonlocking_formatter_optimization<std::basic_string<CharT>>);
  static_assert(std::enable_nonlocking_formatter_optimization<std::basic_string_view<CharT>>);

  static_assert(std::enable_nonlocking_formatter_optimization<bool>);

  static_assert(std::enable_nonlocking_formatter_optimization<signed char>);
  static_assert(std::enable_nonlocking_formatter_optimization<signed short>);
  static_assert(std::enable_nonlocking_formatter_optimization<signed int>);
  static_assert(std::enable_nonlocking_formatter_optimization<signed long>);
  static_assert(std::enable_nonlocking_formatter_optimization<signed long long>);
#ifndef TEST_HAS_NO_INT128
  static_assert(std::enable_nonlocking_formatter_optimization<__int128_t>);
#endif

  static_assert(std::enable_nonlocking_formatter_optimization<unsigned char>);
  static_assert(std::enable_nonlocking_formatter_optimization<unsigned short>);
  static_assert(std::enable_nonlocking_formatter_optimization<unsigned int>);
  static_assert(std::enable_nonlocking_formatter_optimization<unsigned long>);
  static_assert(std::enable_nonlocking_formatter_optimization<unsigned long long>);
#ifndef TEST_HAS_NO_INT128
  static_assert(std::enable_nonlocking_formatter_optimization<__uint128_t>);
#endif

  static_assert(std::enable_nonlocking_formatter_optimization<float>);
  static_assert(std::enable_nonlocking_formatter_optimization<double>);
  static_assert(std::enable_nonlocking_formatter_optimization<long double>);

  static_assert(std::enable_nonlocking_formatter_optimization<std::nullptr_t>);
  static_assert(std::enable_nonlocking_formatter_optimization<void*>);
  static_assert(std::enable_nonlocking_formatter_optimization<const void*>);
}

// Tests for P1361 Integration of chrono with text formatting
//
// Some tests are commented out since these types haven't been implemented in
// chrono yet. After P1361 has been implemented these formatters should be all
// enabled.
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

  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::microseconds>);

  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::sys_time<std::chrono::microseconds>>);
  //static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::utc_time<std::chrono::microseconds>>);
  //static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::tai_time<std::chrono::microseconds>>);
  //static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::gps_time<std::chrono::microseconds>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::file_time<std::chrono::microseconds>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::local_time<std::chrono::microseconds>>);

  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::day>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::month>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::year>);

  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::weekday>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::weekday_indexed>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::weekday_last>);

  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::month_day>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::month_day_last>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::month_weekday>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::month_weekday_last>);

  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::year_month>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::year_month_day>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::year_month_day_last>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::year_month_weekday>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::year_month_weekday_last>);

  static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::hh_mm_ss<std::chrono::microseconds>>);

  //static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::sys_info>);
  //static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::local_info>);

  //static_assert(!std::enable_nonlocking_formatter_optimization<std::chrono::zoned_time>);

#endif // TEST_HAS_NO_LOCALIZATION
}

// Tests for P1636 Formatters for library types
//
// The paper hasn't been voted in so currently all formatters are disabled.
// Note the paper has been abandoned, the types are kept since other papers may
// introduce these formatters.
void test_P1636() {
#ifndef TEST_HAS_NO_THREADS
  static_assert(!std::enable_nonlocking_formatter_optimization<std::thread::id>);
#endif
}

template <class Vector>
void test_P2286_vector_bool() {
  static_assert(!std::enable_nonlocking_formatter_optimization<Vector>);
  static_assert(!std::enable_nonlocking_formatter_optimization<typename Vector::reference>);

  // The const_reference shall be a bool.
  // However libc++ uses a __bit_const_reference<vector> when
  // _LIBCPP_ABI_BITSET_VECTOR_BOOL_CONST_SUBSCRIPT_RETURN_BOOL is defined.
  static_assert(!std::enable_nonlocking_formatter_optimization<const Vector&>);
  static_assert(!std::enable_nonlocking_formatter_optimization<typename Vector::const_reference>);
}

// Tests for P2286 Formatting ranges
void test_P2286() {
  static_assert(!std::enable_nonlocking_formatter_optimization<std::array<int, 42>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::vector<int>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::deque<int>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::forward_list<int>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::list<int>>);

  static_assert(!std::enable_nonlocking_formatter_optimization<std::set<int>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::map<int, int>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::multiset<int>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::multimap<int, int>>);

  static_assert(!std::enable_nonlocking_formatter_optimization<std::unordered_set<int>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::unordered_map<int, int>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::unordered_multiset<int>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::unordered_multimap<int, int>>);

  static_assert(!std::enable_nonlocking_formatter_optimization<std::stack<int>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::queue<int>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::priority_queue<int>>);

  static_assert(!std::enable_nonlocking_formatter_optimization<std::span<int>>);

  static_assert(!std::enable_nonlocking_formatter_optimization<std::valarray<int>>);

  static_assert(!std::enable_nonlocking_formatter_optimization<std::pair<int, int>>);
  static_assert(!std::enable_nonlocking_formatter_optimization<std::tuple<int>>);

  test_P2286_vector_bool<std::vector<bool>>();
  test_P2286_vector_bool<std::vector<bool, std::allocator<bool>>>();
  test_P2286_vector_bool<std::vector<bool, min_allocator<bool>>>();
}

// The trait does not care about whether the type is formattable, obviously the
// trait for non formattable types are not used.
struct not_formattable_nonlocking_disabled {};
static_assert(!std::enable_nonlocking_formatter_optimization<not_formattable_nonlocking_disabled>);

struct not_formattable_nonlocking_enabled {};
template <>
inline constexpr bool std::enable_nonlocking_formatter_optimization<not_formattable_nonlocking_enabled> = true;
static_assert(std::enable_nonlocking_formatter_optimization<not_formattable_nonlocking_enabled>);

void test() {
  test_P0645<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_P0645<wchar_t>();
#endif
  test_P1361();
  test_P1636();
  test_P2286();
}
