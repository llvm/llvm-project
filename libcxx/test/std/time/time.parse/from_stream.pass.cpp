//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// XFAIL: availability-tzdb-missing && !libcpp-has-no-experimental-tzdb && !no-tzdb && !no-filesystem

// <chrono>
// from_stream overloads for calendar types, durations, and time points.

#include <cassert>
#include <chrono>
#include <initializer_list>
#include <limits>
#include <locale>
#include <ratio>
#include <sstream>
#include <string>

#include "make_string.h"
#include "test_macros.h"

#define STR(S) MAKE_STRING(CharT, S)

template <class CharT, class T>
void check(const std::basic_string<CharT>& input, const std::basic_string<CharT>& format, T expected) {
  std::basic_istringstream<CharT> stream(input);
  stream.imbue(std::locale::classic());
  T result{};
  std::chrono::from_stream(stream, format.c_str(), result);
  assert(!stream.fail());
  assert(result == expected);
}

template <class T, class CharT>
void check_failure(const std::basic_string<CharT>& input, const std::basic_string<CharT>& format) {
  std::basic_istringstream<CharT> stream(input);
  stream.imbue(std::locale::classic());
  T result{};
  std::chrono::from_stream(stream, format.c_str(), result);
  assert(stream.fail());
}

template <class CharT>
void test_calendar_types() {
  using namespace std::chrono;
  check(STR("15"), STR("%d"), day{15});
  check(STR(" 15"), STR("%n%e"), day{15});
  // Zone fields are consumed even when their output pointers are null.
  check(STR("15 UTC +0130"), STR("%d %Z %z"), day{15});
  check_failure<day>(STR("02-15"), STR("%m-%d"));
  check_failure<day>(STR("15 00"), STR("%d %H"));
  check_failure<day>(STR("32"), STR("%d"));

  check(STR("02"), STR("%m"), February);
  check(STR("Feb"), STR("%b"), February);
  check_failure<month>(STR("13"), STR("%m"));

  check(STR("2026"), STR("%Y"), 2026y);

  check(STR("Mon"), STR("%a"), Monday);
  check(STR("1"), STR("%u"), Monday);
  check_failure<weekday>(STR("7"), STR("%w"));

  check(STR("02-15"), STR("%m-%d"), February / 15);
  check(STR("02-29"), STR("%m-%d"), February / 29);
  check(STR("32"), STR("%j"), February / 1);
  check_failure<month_day>(STR("60"), STR("%j"));
  check_failure<month_day>(STR("02-30"), STR("%m-%d"));
  check_failure<month_day>(STR("02"), STR("%m"));

  check(STR("2026-02"), STR("%Y-%m"), 2026y / February);
  check_failure<year_month>(STR("2026-13"), STR("%Y-%m"));

  const year_month_day date = 2026y / February / 15;
  check(STR("2026-02-15"), STR("%F"), date);
  check(STR("2026 046"), STR("%Y %j"), date);
  check(STR("2026-W07-7"), STR("%G-W%V-%u"), date);
  check(STR("68 69-W01-1"), STR("%y %g-W%V-%u"), 2068y / December / 31);
  check(STR("-32768-W53-6"), STR("%6G-W%V-%u"), (year::min)() / January / 1);
  check(STR("2026 07 0"), STR("%Y %U %w"), date);
  check(STR("2026 06 7"), STR("%Y %W %u"), date);
  check(STR("2026-02-15 Sun"), STR("%F %a"), date);
  check_failure<year_month_day>(STR("2026-02-15 Mon"), STR("%F %a"));
  check_failure<year_month_day>(STR("2026-02-30"), STR("%F"));
  check_failure<year_month_day>(STR("2026-02"), STR("%Y-%m"));
  check_failure<year_month_day>(STR("2021-W53-5"), STR("%G-W%V-%u"));
}

template <class CharT>
void test_format() {
  using namespace std::chrono;
  const sys_seconds date      = sys_days{2026y / July / 20};
  const sys_seconds date_time = date + 13h + 45min + 30s;

  // Individual and compound directives.
  check(STR("2026-07-20 13:45:30"), STR("%Y-%m-%d %H:%M:%S"), date_time);
  check(STR("2026-07-7"), STR("%Y-%m-%e"), sys_seconds{sys_days{2026y / July / 7}});
  check(STR("2026-07-20 13:45:30"), STR("%F %T"), date_time);
  check(STR("2026-07-20 13:45"), STR("%F %R"), date + 13h + 45min);
  check(STR("07/20/26"), STR("%D"), date);
  check(STR("30"), STR("%S"), 30s);

  // Named fields and locale-specific date/time formats in the classic locale.
  check(STR("Monday"), STR("%A"), Monday);
  check(STR("July"), STR("%B"), July);
  check(STR("Jul"), STR("%h"), July);
  check(STR("Mon Jul 20 13:45:30 2026"), STR("%c"), date_time);
  check_failure<sys_seconds>(STR("Tue Jul 20 13:45:30 2026"), STR("%c"));
  check(STR("07/20/26"), STR("%x"), date);

  // A width on %F applies only to %Y.
  check(STR("002026-07-20"), STR("%6F"), date);
  check(STR("26-W30-1"), STR("%g-W%V-%u"), date);

  // Literals and whitespace, including a match of zero whitespace characters.
  check(STR("2026-07-20%"), STR("%F%%"), date);
  check(STR("2026\t07\t20"), STR("%Y%t%m%t%d"), date);
  check(STR("   2026-07-20"), STR(" %F"), date);
  check(STR("2026-07-20"), STR(" %Y-%m-%d"), date);

  check_failure<sys_seconds>(STR("2026/07/20"), STR("%Y-%m-%d"));
  check_failure<sys_seconds>(STR("13:45:30"), STR("%H:%M:%S"));
  check_failure<sys_seconds>(STR("2026-07-xx"), STR("%Y-%m-%d"));
  check_failure<sys_seconds>(STR("2026-07- 7"), STR("%Y-%m-%e"));
}

template <class CharT>
void test_width_overflow() {
  using namespace std::chrono;
  std::basic_ostringstream<CharT> format;
  format << CharT('%') << std::numeric_limits<unsigned>::max() << STR("0S");

  // Reject an overflowing format width before reading the input.
  std::basic_istringstream<CharT> stream{STR("1.25!")};
  milliseconds result{};
  from_stream(stream, format.str().c_str(), result);
  assert(stream.fail());
  stream.clear();
  assert(stream.peek() == CharT('1'));
}

template <class CharT>
void test_date_consistency() {
  using namespace std::chrono;
  const year_month_day feb1 = 2026y / February / 1;
  const year_month_day jan1 = 2021y / January / 1;

  // An ordinal date must agree with the supplied month.
  check(STR("2026 032 02"), STR("%Y %j %m"), feb1);
  check_failure<year_month_day>(STR("2026 032 03"), STR("%Y %j %m"));

  // An ISO week date must agree with the calendar year, even across a year boundary.
  check(STR("2020-W53-5 2021"), STR("%G-W%V-%u %Y"), jan1);
  check_failure<year_month_day>(STR("2020-W53-5 2020"), STR("%G-W%V-%u %Y"));
  check(STR("1926-W01-1 26"), STR("%G-W%V-%u %g"), 1926y / January / 4);
}

template <class CharT>
void test_modifier_E() {
  using namespace std::chrono;
  const sys_seconds date = sys_days{2026y / July / 20};

  check(STR("Mon Jul 20 13:45:30 2026"), STR("%Ec"), date + 13h + 45min + 30s);
  check(STR("07/20/26"), STR("%Ex"), date);

  // Numeric years in the classic locale.
  for (const auto& format : {STR("%EC %Ey-%m-%d"), STR("%EC %EY-%m-%d")}) {
    std::basic_istringstream<CharT> stream(format == STR("%EC %Ey-%m-%d") ? STR("20 26-07-20") : STR("20 2026-07-20"));
    stream.imbue(std::locale::classic());
    sys_seconds result{};
    from_stream(stream, format.c_str(), result);
    assert(!stream.fail());
    assert(result == date);
  }

  // Parse a century and a UTC offset.
  {
    std::basic_istringstream<CharT> stream(STR("20 26-07-20 +4:30"));
    stream.imbue(std::locale::classic());
    sys_seconds result{};
    from_stream(stream, STR("%EC %y-%m-%d %Ez").c_str(), result);
    assert(!stream.fail());
    assert(result == date - 4h - 30min);
  }
}

template <class CharT>
void test_modifier_O() {
  using namespace std::chrono;
  const sys_seconds date = sys_days{2026y / July / 20};

  check(STR("2026-07-20"), STR("%Y-%m-%Oe"), date);

  // Parse numeric fields, including twelve o'clock with PM.
  {
    std::basic_istringstream<CharT> stream(STR("26-07-20 13:45 1!"));
    stream.imbue(std::locale::classic());
    sys_seconds result{};
    from_stream(stream, STR("%Oy-%Om-%Od %OH:%OM %Ow").c_str(), result);
    assert(!stream.fail());
    assert(result == date + 13h + 45min);
    assert(stream.peek() == CharT('!'));
  }
  {
    std::basic_istringstream<CharT> stream(STR("2026-07-20 12 PM"));
    stream.imbue(std::locale::classic());
    sys_seconds result{};
    from_stream(stream, STR("%F %OI %p").c_str(), result);
    assert(!stream.fail());
    assert(result == date + 12h);
  }

  // %OS, %OU and %OW parse numerically; %Oz uses the offset parser.
  {
    std::basic_istringstream<CharT> stream(STR("2026-07-20 13:45:30.125 +4:30"));
    stream.imbue(std::locale::classic());
    sys_time<milliseconds> result{};
    from_stream(stream, STR("%F %H:%M:%OS %Oz").c_str(), result);
    assert(!stream.fail());
    assert(result == date + 13h + 45min + 30s + 125ms - 4h - 30min);
  }
  for (const auto& format : {STR("%Y %OU %w"), STR("%Y %OW %w")}) {
    std::basic_istringstream<CharT> stream(STR("2026 29 1"));
    stream.imbue(std::locale::classic());
    sys_seconds result{};
    from_stream(stream, format.c_str(), result);
    assert(!stream.fail());
    assert(result == date);
  }
}

template <class CharT>
struct comma_numpunct : std::numpunct<CharT> {
  CharT do_decimal_point() const override { return CharT(','); }
};

template <class CharT>
void test_fractional_seconds() {
  using namespace std::chrono;
  check(STR("1.25"), STR("%S"), 1250ms);
  check(STR(".5"), STR("%S"), 500ms);
  check(STR("1."), STR("%S"), 1000ms);
  check(STR(".5!"), STR("%2S!"), 500ms);
  // Floating-point representations allow fractions even with a whole-second period.
  check(STR("12.125s"), STR("%Ss"), duration<double>{12.125});
  // An explicit width controls consumption independently of the target precision.
  check(STR("1.2345!"), STR("%6S!"), 1234ms);

  // An integer seconds target leaves the decimal point unread, regardless of width.
  for (const auto& format : {STR("%S"), STR("%6S")}) {
    std::basic_istringstream<CharT> stream(STR("1.25"));
    stream.imbue(std::locale::classic());
    duration<int> result{};
    from_stream(stream, format.c_str(), result);
    assert(!stream.fail());
    assert(result == 1s);
    assert(stream.peek() == CharT('.'));
  }
  {
    std::basic_istringstream<CharT> stream(STR("1,25!"));
    stream.imbue(std::locale(std::locale::classic(), new comma_numpunct<CharT>));
    milliseconds result{};
    from_stream(stream, STR("%S").c_str(), result);
    assert(!stream.fail());
    assert(result == 1250ms);
    assert(stream.peek() == CharT('!'));
  }
  {
    // The width includes the decimal point and limits the fractional digits.
    std::basic_istringstream<CharT> stream(STR("1.234"));
    stream.imbue(std::locale::classic());
    milliseconds result{};
    from_stream(stream, STR("%3S").c_str(), result);
    assert(!stream.fail());
    assert(result == 1200ms);
    assert(stream.peek() == CharT('3'));
  }
}

template <class CharT>
void test_time_points() {
  using namespace std::chrono;
  const sys_seconds expected = sys_days{2026y / July / 20} + 13h + 45min + 30s;
  auto test = [&](auto value) { check(STR("20 26-07-20 01:45:30 PM"), STR("%C %y-%m-%d %I:%M:%S %p"), value); };
  test(expected);
  test(local_seconds{expected.time_since_epoch()});
  test(file_clock::from_sys(expected));

  check(STR("2026-07-20"), STR("%F"), sys_days{2026y / July / 20});
  // A days target rejects time-of-day fields even when the UTC offset would cancel them.
  check_failure<sys_days>(STR("2026-07-20 04:00 +0400"), STR("%F %R %z"));
  check(STR("1969-12-31 23:59:59.5"), STR("%F %T"), sys_time<milliseconds>{-500ms});
  check(STR("1970-01-01 00:00:12.125!"), STR("%F %T!"), sys_time<duration<double>>{duration<double>{12.125}});
  check(STR("1970-01-01 00:00:01"), STR("%F %T"), sys_time<duration<long long, std::atto>>{1s});

#if !defined(TEST_HAS_NO_EXPERIMENTAL_TZDB) && !defined(TEST_HAS_NO_TIME_ZONE_DATABASE) &&                             \
    !defined(TEST_HAS_NO_FILESYSTEM)
  // Combine the date and time of day before rounding to a unit that does not divide a day.
  using SevenSeconds = duration<long long, std::ratio<7>>;
  check(STR("1958-01-02 00:02"), STR("%F %R"), tai_time<SevenSeconds>{SevenSeconds{12360}});
#endif
}

#if !defined(TEST_HAS_NO_EXPERIMENTAL_TZDB) && !defined(TEST_HAS_NO_TIME_ZONE_DATABASE) &&                             \
    !defined(TEST_HAS_NO_FILESYSTEM)
template <class CharT>
void test_utc_leap_seconds() {
  using namespace std::chrono;
  const utc_seconds leap = utc_clock::from_sys(sys_days{2017y / January / 1}) - 1s;
  check(STR("2016-12-31 23:59:60"), STR("%F %T"), leap);
  check(STR("2017-01-01 00:59:60.5 +0100"), STR("%F %T %z"), leap + 500ms);
  check_failure<utc_seconds>(STR("2026-07-20 12:00:60"), STR("%F %T"));
}
#endif

template <class CharT>
void test_year_fields() {
  using namespace std::chrono;
  check(STR("+123"), STR("%4Y"), year{123});
  check(STR("-123"), STR("%4Y"), year{-123});
  check(STR("0"), STR("%Y"), year{0});

  // Two-digit years use a default century unless %C or %Y supplies one.
  check(STR("68"), STR("%y"), year{2068});
  check(STR("69"), STR("%y"), year{1969});
  check(STR("20 26"), STR("%C %y"), 2026y);
  check(STR("-20 76"), STR("%3C %y"), year{-1976});
  check(STR("2026 20"), STR("%Y %C"), 2026y);
  check(STR("1926 26"), STR("%Y %y"), 1926y);
  check(STR("1926-07-20 26"), STR("%F %y"), 1926y / July / 20);

  // A sign consumes one character of the field's width.
  {
    std::basic_istringstream<CharT> stream(STR("+123"));
    stream.imbue(std::locale::classic());
    year result{};
    from_stream(stream, STR("%3Y").c_str(), result);
    assert(!stream.fail());
    assert(result == year{12});
    assert(stream.peek() == CharT('3'));
  }

  check_failure<year>(STR("100"), STR("%3y"));
  check_failure<year>(STR("+1"), STR("%1Y"));
  check_failure<year>(STR("2026 25"), STR("%Y %y"));
  check_failure<year>(STR("20"), STR("%C"));
}

template <class CharT>
void test_duration() {
  using namespace std::chrono;
  // Parse time-of-day and day-count components.
  for (const auto& format : {STR("%T"), STR("%X"), STR("%EX")})
    check(STR("01:30:00"), format, 5400s);
  check(STR("01:30:00 AM"), STR("%r"), 5400s);
  check(STR("PM 01:30"), STR("%p %I:%M"), 810min);
  check(STR("2"), STR("%j"), days{2});
  check(STR("2 01:30"), STR("%j %R"), 48h + 90min);
  check(STR("23:59:59"), STR("%T"), 23h + 59min + 59s);

  check_failure<seconds>(STR("15 01"), STR("%d %H"));
  check_failure<hours>(STR("01:30"), STR("%R"));
  check_failure<minutes>(STR("PM"), STR("%p"));
  check_failure<minutes>(STR("60"), STR("%M"));
  check_failure<seconds>(STR("60"), STR("%S"));
}

template <class CharT>
void test_hours() {
  using namespace std::chrono;
  // %H can disambiguate %I without %p, in either input order.
  check(STR("13 01 PM"), STR("%H %I %p"), 13h);
  check(STR("13 01"), STR("%H %I"), 13h);
  check(STR("01 13"), STR("%I %H"), 13h);
  check(STR("00 12"), STR("%H %I"), 0h);
  check(STR("12 12"), STR("%H %I"), 12h);

  check_failure<minutes>(STR("14 01"), STR("%H %I"));
  check_failure<minutes>(STR("13 AM"), STR("%H %p"));
  check_failure<hours>(STR("24"), STR("%H"));
  check_failure<minutes>(STR("13 PM"), STR("%I %p"));
}

template <class CharT>
void test_duration_conversion() {
  using namespace std::chrono;

  check(STR("42"), STR("%S"), duration<unsigned>{42});
  check(STR("1.25"), STR("%S"), duration<double, std::milli>{1250});
  // A 1.5-second integer tick is coarser than the seconds field.
  check_failure<duration<int, std::ratio<3, 2>>>(STR("1.5"), STR("%S"));
}

template <class CharT>
void test_offsets() {
  using namespace std::chrono;
  const sys_seconds date = sys_days{2026y / July / 20};
  check(STR("2026-07-20 04"), STR("%F %z"), date - 4h);
  check(STR("2026-07-20 0430"), STR("%F %z"), date - 4h - 30min);
  check(STR("2026-07-20 -0130"), STR("%F %z"), date + 90min);
  // Offset minutes are not clock-minute fields restricted to 0-59.
  check(STR("2026-07-20 +0160"), STR("%F %z"), date - 120min);

  check(STR("2026-07-20 4"), STR("%F %Ez"), date - 4h);
  check(STR("2026-07-20 +4:30"), STR("%F %Ez"), date - 4h - 30min);
  check(STR("2026-07-20 +4:30"), STR("%F %Oz"), date - 4h - 30min);
  check(STR("2026-07-20 -4:30"), STR("%F %Ez"), date + 4h + 30min);

  // Basic hours and any supplied minutes require two digits.
  check_failure<sys_seconds>(STR("2026-07-20 019"), STR("%F %z"));
  check_failure<sys_seconds>(STR("2026-07-20 4:9"), STR("%F %Ez"));
  check_failure<sys_seconds>(STR("2026-07-20 4"), STR("%F %z"));
}

template <class CharT>
void test_abbrev() {
  using namespace std::chrono;
  std::basic_istringstream<CharT> stream(STR("2026-07-20 a_1/+B-2!"));
  stream.imbue(std::locale::classic());
  sys_days result{};
  std::basic_string<CharT> abbrev = STR("original");
  from_stream(stream, STR("%F %Z").c_str(), result, &abbrev);
  assert(!stream.fail());
  assert(result == sys_days{2026y / July / 20});
  assert(abbrev == STR("a_1/+B-2"));
  assert(stream.peek() == CharT('!'));
}

template <class CharT>
void test() {
  test_calendar_types<CharT>();
  test_format<CharT>();
  test_width_overflow<CharT>();
  test_date_consistency<CharT>();
  test_modifier_E<CharT>();
  test_modifier_O<CharT>();
  test_fractional_seconds<CharT>();
  test_year_fields<CharT>();
  test_duration<CharT>();
  test_hours<CharT>();
  test_duration_conversion<CharT>();
  test_time_points<CharT>();
#if !defined(TEST_HAS_NO_EXPERIMENTAL_TZDB) && !defined(TEST_HAS_NO_TIME_ZONE_DATABASE) &&                             \
    !defined(TEST_HAS_NO_FILESYSTEM)
  test_utc_leap_seconds<CharT>();
#endif
  test_offsets<CharT>();
  test_abbrev<CharT>();
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  return 0;
}
