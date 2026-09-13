//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>

// template <class _Tm, class _Date>
// _LIBCPP_HIDE_FROM_ABI _Tm __convert_to_tm(const _Date& __date, chrono::weekday __weekday)
//
// template <class _Tm, class _ChronoT>
// _LIBCPP_HIDE_FROM_ABI _Tm __convert_to_tm(const _ChronoT& __value)

// Most of the code is tested indirectly in the chrono formatters. This tests
// the hour overflow and setting tm_zone to "UTC" when supported.

#include <__chrono/convert_to_tm.h>
#include <chrono>
#include <cassert>
#include <ctime>
#include <format>
#include <string_view>

#include "test_macros.h"

// libc++ uses a long as representation in std::chrono::hours.
// std::tm uses an int for its integral members. The overflow in the hour
// conversion can only occur on platforms where sizeof(long) > sizeof(int).
// Instead emulate this error by using a "tm" with shorts.
// (The function is already templated, so this is quite easy to do.)
struct minimal_short_tm {
  short tm_sec;
  short tm_min;
  short tm_hour;
};

struct tm_with_const_zone {
  int tm_sec;
  int tm_min;
  int tm_hour;
  int tm_mday;
  int tm_mon;
  int tm_year;
  int tm_wday;
  int tm_yday;
  int tm_isdst;
  const char* tm_zone;
};

struct tm_with_char_zone {
  int tm_sec;
  int tm_min;
  int tm_hour;
  int tm_mday;
  int tm_mon;
  int tm_year;
  int tm_wday;
  int tm_yday;
  int tm_isdst;
  char* tm_zone;
};

static void test_hour_overflow() {
  { // Test with the maximum number of hours that fit in a short.
    std::chrono::hh_mm_ss time{std::chrono::hours{32767}};
    minimal_short_tm result = std::__convert_to_tm<minimal_short_tm>(time);
    assert(result.tm_sec == 0);
    assert(result.tm_min == 0);
    assert(result.tm_hour == 32767);
  }

#ifndef TEST_HAS_NO_EXCEPTIONS
  { // Test above the maximum number of hours that fit in a short.
    std::chrono::hh_mm_ss time{std::chrono::hours{32768}};
    try {
      TEST_IGNORE_NODISCARD std::__convert_to_tm<minimal_short_tm>(time);
      assert(false);
    } catch ([[maybe_unused]] const std::format_error& e) {
      LIBCPP_ASSERT(e.what() == std::string_view("Formatting hh_mm_ss, encountered an hour overflow"));
      return;
    }
    assert(false);
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

template <class Tm = std::tm>
static void test_std_tm() {
  using namespace std::chrono_literals;

  {
    auto date    = std::chrono::year_month_day{2023y, std::chrono::January, 1d};
    auto weekday = std::chrono::Sunday;
    Tm result    = std::__convert_to_tm<Tm>(date, weekday);
    if constexpr (requires { result.tm_zone; }) {
      assert(result.tm_zone != nullptr);
      assert(std::string_view(result.tm_zone) == "UTC");
    }
  }
  {
    auto date = std::chrono::year_month_day{2023y, std::chrono::January, 1d};
    Tm result = std::__convert_to_tm<Tm>(date);
    if constexpr (requires { result.tm_zone; }) {
      assert(result.tm_zone != nullptr);
      assert(std::string_view(result.tm_zone) == "UTC");
    }
  }
}

static void test_tm_zone() {
  using namespace std::chrono_literals;

  // 1. Test __convert_to_tm(date, weekday) overload with tm_with_const_zone and tm_with_char_zone
  {
    auto date                 = std::chrono::year_month_day{2023y, std::chrono::January, 1d};
    auto weekday              = std::chrono::Sunday;
    tm_with_const_zone result = std::__convert_to_tm<tm_with_const_zone>(date, weekday);
    assert(result.tm_zone != nullptr);
    assert(std::string_view(result.tm_zone) == "UTC");
  }
  {
    auto date                = std::chrono::year_month_day{2023y, std::chrono::January, 1d};
    auto weekday             = std::chrono::Sunday;
    tm_with_char_zone result = std::__convert_to_tm<tm_with_char_zone>(date, weekday);
    assert(result.tm_zone != nullptr);
    assert(std::string_view(result.tm_zone) == "UTC");
  }

  // 2. Test __convert_to_tm(value) overload with tm_with_const_zone and tm_with_char_zone
  {
    auto date                 = std::chrono::year_month_day{2023y, std::chrono::January, 1d};
    tm_with_const_zone result = std::__convert_to_tm<tm_with_const_zone>(date);
    assert(result.tm_zone != nullptr);
    assert(std::string_view(result.tm_zone) == "UTC");
  }
  {
    auto date                = std::chrono::year_month_day{2023y, std::chrono::January, 1d};
    tm_with_char_zone result = std::__convert_to_tm<tm_with_char_zone>(date);
    assert(result.tm_zone != nullptr);
    assert(std::string_view(result.tm_zone) == "UTC");
  }
  {
    std::chrono::sys_days tp  = std::chrono::year_month_day{2023y, std::chrono::January, 1d};
    tm_with_const_zone result = std::__convert_to_tm<tm_with_const_zone>(tp);
    assert(result.tm_zone != nullptr);
    assert(std::string_view(result.tm_zone) == "UTC");
  }
  {
    std::chrono::sys_days tp = std::chrono::year_month_day{2023y, std::chrono::January, 1d};
    tm_with_char_zone result = std::__convert_to_tm<tm_with_char_zone>(tp);
    assert(result.tm_zone != nullptr);
    assert(std::string_view(result.tm_zone) == "UTC");
  }
  {
    std::chrono::hh_mm_ss time{std::chrono::hours{12}};
    tm_with_const_zone result = std::__convert_to_tm<tm_with_const_zone>(time);
    assert(result.tm_zone != nullptr);
    assert(std::string_view(result.tm_zone) == "UTC");
  }
  {
    std::chrono::hh_mm_ss time{std::chrono::hours{12}};
    tm_with_char_zone result = std::__convert_to_tm<tm_with_char_zone>(time);
    assert(result.tm_zone != nullptr);
    assert(std::string_view(result.tm_zone) == "UTC");
  }

  // 3. Test with std::tm (if std::tm has tm_zone on the target platform)
  test_std_tm();

  // 4. Test that a type without tm_zone continues to work
  {
    auto date    = std::chrono::year_month_day{2023y, std::chrono::January, 1d};
    auto weekday = std::chrono::Sunday;
    struct tm_without_zone {
      int tm_sec;
      int tm_min;
      int tm_hour;
      int tm_mday;
      int tm_mon;
      int tm_year;
      int tm_wday;
      int tm_yday;
      int tm_isdst;
    };
    tm_without_zone result1 = std::__convert_to_tm<tm_without_zone>(date, weekday);
    assert(result1.tm_year == 123);
    assert(result1.tm_mon == 0);
    assert(result1.tm_mday == 1);

    tm_without_zone result2 = std::__convert_to_tm<tm_without_zone>(date);
    assert(result2.tm_year == 123);
    assert(result2.tm_mon == 0);
    assert(result2.tm_mday == 1);
  }
}

int main(int, char**) {
  test_hour_overflow();
  test_tm_zone();

  return 0;
}
