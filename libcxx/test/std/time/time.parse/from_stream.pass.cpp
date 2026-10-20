//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-localization
// XFAIL: availability-tzdb-missing && !libcpp-has-no-experimental-tzdb && !no-tzdb && !no-filesystem

// <chrono>
//
// Tested from_stream overloads (C++20):
//
// template<class charT, class traits, class Rep, class Period, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//     from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//                 duration<Rep, Period>& d, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//                 minutes* offset = nullptr);
//
// template<class charT, class traits, class Duration, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//     from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//                 sys_time<Duration>& tp, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//                 minutes* offset = nullptr);
//
// template<class charT, class traits, class Duration, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//     from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//                 utc_time<Duration>& tp, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//                 minutes* offset = nullptr);
//
// template<class charT, class traits, class Duration, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//     from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//                 tai_time<Duration>& tp, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//                 minutes* offset = nullptr);
//
// template<class charT, class traits, class Duration, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//     from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//                 gps_time<Duration>& tp, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//                 minutes* offset = nullptr);
//
// template<class charT, class traits, class Duration, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//     from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//                 file_time<Duration>& tp, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//                 minutes* offset = nullptr);
//
// template<class charT, class traits, class Duration, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//     from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//                 local_time<Duration>& tp, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//                 minutes* offset = nullptr);
//
// template<class charT, class traits, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//     from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//                 day& d, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//                 minutes* offset = nullptr);
//
// template<class charT, class traits, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//     from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//                 month& m, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//                 minutes* offset = nullptr);
//
// template<class charT, class traits, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//     from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//                 year& y, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//                 minutes* offset = nullptr);
//
// template<class charT, class traits, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//     from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//                 weekday& wd, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//                 minutes* offset = nullptr);
//
// template<class charT, class traits, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//     from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//                 month_day& md, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//                 minutes* offset = nullptr);
//
// template<class charT, class traits, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//     from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//                 year_month& ym, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//                 minutes* offset = nullptr);
//
// template<class charT, class traits, class Alloc = allocator<charT>>
//   basic_istream<charT, traits>&
//     from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//                 year_month_day& ymd, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//                 minutes* offset = nullptr);

#include <cassert>
#include <chrono>
#include <initializer_list>
#include <locale>
#include <ratio>
#include <sstream>
#include <string>

#include "make_string.h"
#include "test_macros.h"

using namespace std::chrono;

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
void check_failure(const std::basic_string<CharT>& input, const std::basic_string<CharT>& format, T initial = T{}) {
  std::basic_istringstream<CharT> stream(input);
  stream.imbue(std::locale::classic());
  T result = initial;
  std::chrono::from_stream(stream, format.c_str(), result);
  assert(stream.fail());
  assert(result == initial);
}

// Test parsing day, month, year, weekday, month_day, year_month, and year_month_day.
// Cover numeric and named calendar fields, invalid values, missing fields,
// and fields that the target type cannot represent.
template <class CharT>
void test_calendar_types() {
  // Parse a numeric day of the month.
  check(STR("15"), STR("%d"), day{15});
  // Parse the day with %e, which is equivalent to %d.
  check(STR("15"), STR("%e"), day{15});
  // A day cannot represent a month field.
  check_failure<day>(STR("02-15"), STR("%m-%d"));
  // A day cannot represent an hour field, even when its value is zero.
  check_failure<day>(STR("15 00"), STR("%d %H"));
  // Day 0 is below the valid range.
  check_failure<day>(STR("0"), STR("%d"));
  // Day 32 is outside the valid range; preserve the original day on failure.
  check_failure<day>(STR("32"), STR("%d"), day{7});

  // Parse a numeric month.
  check(STR("02"), STR("%m"), February);
  // Parse an abbreviated month name in the classic locale.
  check(STR("Feb"), STR("%b"), February);
  // Parse a full month name.
  check(STR("July"), STR("%B"), July);
  // %B also accepts an abbreviated month name, ignoring case.
  check(STR("jUl"), STR("%B"), July);
  // %h is equivalent to %b.
  check(STR("Jul"), STR("%h"), July);
  // Month 13 is outside the valid range.
  check_failure<month>(STR("13"), STR("%m"));

  // Parse a calendar year.
  check(STR("2026"), STR("%Y"), 2026y);

  // Parse an abbreviated weekday name in the classic locale.
  check(STR("Mon"), STR("%a"), Monday);
  // %a also accepts a full weekday name, ignoring case.
  check(STR("mOnDaY"), STR("%a"), Monday);
  // Parse a full weekday name.
  check(STR("Monday"), STR("%A"), Monday);
  // ISO weekday numbering assigns 1 to Monday.
  check(STR("1"), STR("%u"), Monday);
  // Unlike %u, %w only accepts weekday numbers from 0 to 6.
  check_failure<weekday>(STR("7"), STR("%w"));

  // Parse a month and day without a year.
  check(STR("02-15"), STR("%m-%d"), February / 15);
  // February 29 is valid for month_day because no year is specified.
  check(STR("02-29"), STR("%m-%d"), February / 29);
  // February 30 is invalid in every year.
  check_failure<month_day>(STR("02-30"), STR("%m-%d"));
  // A month alone does not supply the required day.
  check_failure<month_day>(STR("02"), STR("%m"));

  // Parse a year and month without a day.
  check(STR("2026-02"), STR("%Y-%m"), 2026y / February);
  // Reject an invalid month in a year_month result.
  check_failure<year_month>(STR("2026-13"), STR("%Y-%m"));

  // Parse a complete calendar date with %F.
  check(STR("2026-02-15"), STR("%F"), 2026y / February / 15);
  // %e accepts a single-digit day without padding.
  check(STR("2026-07-7"), STR("%Y-%m-%e"), 2026y / July / 7);
  // Reject an invalid day in a complete calendar date.
  check_failure<year_month_day>(STR("2026-02-30"), STR("%F"));
  // February 29 is valid in a leap year.
  check(STR("2024-02-29"), STR("%F"), 2024y / February / 29);
  // February 29 is invalid in a non-leap year.
  check_failure<year_month_day>(STR("2023-02-29"), STR("%F"));
  // A year and month do not supply the required day.
  check_failure<year_month_day>(STR("2026-02"), STR("%Y-%m"));
}

// Test compound date/time directives against their component fields.
template <class CharT>
void test_compound_specifiers() {
  const sys_days date         = sys_days{2026y / July / 20};
  const sys_seconds date_time = date + 13h + 45min + 30s;

  // Parse the explicit fields represented by %F and %T.
  check(STR("2026-07-20 13:45:30"), STR("%Y-%m-%d %H:%M:%S"), date_time);
  // %F and %T combine the date and time fields.
  check(STR("2026-07-20 13:45:30"), STR("%F %T"), date_time);
  // %R supplies hours and minutes.
  check(STR("2026-07-20 13:45"), STR("%F %R"), sys_time<minutes>{date} + 13h + 45min);
  // %T combines hours, minutes, and seconds into a duration.
  check(STR("01:30:00"), STR("%T"), 5400s);
  // %D combines the month, day, and two-digit year.
  check(STR("07/20/26"), STR("%D"), year_month_day{date});
}

// Test the classic locale's date, time, and date-and-time representations.
template <class CharT>
void test_locale_formats() {
  const year_month_day date   = 2026y / July / 20;
  const sys_seconds date_time = sys_days{date} + 13h + 45min + 30s;

  // Locale-specific date/time formats in the classic locale.
  check(STR("Mon Jul 20 13:45:30 2026"), STR("%c"), date_time);
  // %x parses the classic locale's month/day/two-digit-year representation.
  check(STR("07/20/26"), STR("%x"), date);
  // %X parses the classic locale's hours, minutes, and seconds.
  check(STR("01:30:00"), STR("%X"), 5400s);
  // Parse the classic locale's twelve-hour time with an AM designator.
  check(STR("01:30:00 AM"), STR("%r"), 5400s);
}

// Test literal matching and whitespace consumption in the input.
template <class CharT>
void test_literals_and_whitespace() {
  const year_month_day date = 2026y / July / 20;

  // %% matches a literal percent sign.
  check(STR("2026-07-20%"), STR("%F%%"), date);
  // %t matches a tab between numeric fields.
  check(STR("2026\t07\t20"), STR("%Y%t%m%t%d"), date);
  // %t can match no whitespace before a literal.
  check(STR("X15"), STR("%tX%d"), day{15});
  // A format space matches multiple input spaces.
  check(STR("   2026-07-20"), STR(" %F"), date);
  // A format space can also match no whitespace.
  check(STR("2026-07-20"), STR(" %Y-%m-%d"), date);
  // %n consumes one whitespace character before the day.
  check(STR(" 15"), STR("%n%e"), day{15});
  // %n fails when no whitespace precedes the literal.
  check_failure<day>(STR("X15"), STR("%nX%d"));

  // Literal separators must match.
  check_failure<year_month_day>(STR("2026/07/20"), STR("%Y-%m-%d"));
}

// Test day-of-year and week-based dates, year boundaries, and matching date fields.
template <class CharT>
void test_date_resolution() {
  const year_month_day date = 2026y / February / 15;
  const year_month_day feb1 = 2026y / February / 1;
  const year_month_day jan1 = 2021y / January / 1;

  // Day-of-year 32 is February 1 in both leap and non-leap years.
  check(STR("32"), STR("%j"), February / 1);
  // Day-of-year 60 needs a year to distinguish February 29 from March 1.
  check_failure<month_day>(STR("60"), STR("%j"));
  // Convert a year and day-of-year to a calendar date.
  check(STR("2026 046"), STR("%Y %j"), date);
  // Parse an ordinal date with a matching month.
  check(STR("2026 032 02"), STR("%Y %j %m"), feb1);

  // Convert an ISO week year, week number, and weekday to a calendar date.
  check(STR("2026-W07-7"), STR("%G-W%V-%u"), date);
  // A calendar year supplies context for a two-digit ISO week year.
  check(STR("2026 26-W30-1"), STR("%Y %g-W%V-%u"), 2026y / July / 20);
  // Calendar year 2068 ends in ISO week year 2069, despite %y's century cutoff at 69.
  check(STR("68 69-W01-1"), STR("%y %g-W%V-%u"), 2068y / December / 31);
  // The earliest supported calendar year begins in the preceding ISO week year.
  check(STR("-32768-W53-6"), STR("%6G-W%V-%u"), (year::min)() / January / 1);
  // ISO week years and calendar years can differ across a year boundary.
  check(STR("2020-W53-5 2021"), STR("%G-W%V-%u %Y"), jan1);
  // Match a full ISO week year with its last two digits.
  check(STR("1926-W01-1 26"), STR("%G-W%V-%u %g"), 1926y / January / 4);
  // ISO week year 2021 has no week 53.
  check_failure<year_month_day>(STR("2021-W53-5"), STR("%G-W%V-%u"));

  // Parse a Sunday-based week number with Sunday numbered 0.
  check(STR("2026 07 0"), STR("%Y %U %w"), date);
  // Parse a Monday-based week number with ISO Sunday numbered 7.
  check(STR("2026 06 7"), STR("%Y %W %u"), date);
  // Accept a date accompanied by its matching weekday.
  check(STR("2026-02-15 Sun"), STR("%F %a"), date);
}

// Test E-modified fields in the classic locale, including UTC offsets.
template <class CharT>
void test_modifier_E() {
  const year_month_day date   = 2026y / July / 20;
  const sys_seconds date_time = sys_days{date} + 13h + 45min + 30s;

  // %Ec parses the same date-and-time representation as %c in the classic locale.
  check(STR("Mon Jul 20 13:45:30 2026"), STR("%Ec"), date_time);
  // %Ex parses the same date representation as %x in the classic locale.
  check(STR("07/20/26"), STR("%Ex"), date);
  // %EX parses the same time representation as %X in the classic locale.
  check(STR("01:30:00"), STR("%EX"), 5400s);

  // Numeric years in the classic locale.
  check(STR("20 26-07-20"), STR("%EC %Ey-%m-%d"), date);
  // Match a century with a full year using E-modified fields.
  check(STR("20 2026-07-20"), STR("%EC %EY-%m-%d"), date);

  // Parse a century and a UTC offset.
  check(STR("20 26-07-20 13:45:30 +4:30"), STR("%EC %y-%m-%d %T %Ez"), date_time - 4h - 30min);
}

// Test O-modified numeric fields in the classic locale.
template <class CharT>
void test_modifier_O() {
  const sys_days date = sys_days{2026y / July / 20};

  // %Oe accepts ordinary decimal day digits in the classic locale.
  check(STR("2026-07-20"), STR("%Y-%m-%Oe"), year_month_day{date});

  // Parse numeric fields and leave the trailing character unread.
  {
    std::basic_istringstream<CharT> stream(STR("26-07-20 13:45 1!"));
    stream.imbue(std::locale::classic());
    sys_time<minutes> result{};
    from_stream(stream, STR("%Oy-%Om-%Od %OH:%OM %Ow").c_str(), result);
    assert(!stream.fail());
    assert(result == date + 13h + 45min);
    assert(stream.peek() == CharT('!'));
  }
  // Twelve o'clock with PM denotes noon.
  check(STR("2026-07-20 12 PM"), STR("%F %OI %p"), sys_time<hours>{date} + 12h);

  // Parse fractional seconds with a colon-separated UTC offset.
  check(STR("2026-07-20 13:45:30.125 +4:30"),
        STR("%F %H:%M:%OS %Oz"),
        date + 13h + 45min + 30s + 125ms - 4h - 30min);

  // This date has week number 29 with either Sunday or Monday as the first day.
  for (const auto& format : {STR("%Y %OU %w"), STR("%Y %OW %w")})
    check(STR("2026 29 1"), format, year_month_day{date});
}

template <class CharT>
struct comma_numpunct : std::numpunct<CharT> {
  CharT do_decimal_point() const override { return CharT(','); }
};

// Test fractional-second syntax, decimal punctuation, and target precision.
template <class CharT>
void test_fractional_seconds() {
  // Convert fractional seconds to an integral millisecond count.
  check(STR("1.25"), STR("%S"), 1250ms);
  // Accept a fraction without digits before the decimal point.
  check(STR(".5"), STR("%S"), 500ms);
  // Accept a decimal point without following fractional digits.
  check(STR("1."), STR("%S"), 1000ms);
  // Floating-point representations allow fractions even with a whole-second period.
  check(STR("12.125s"), STR("%Ss"), duration<double>{12.125});

  // With the default width, an integer seconds target leaves the decimal point unread.
  {
    std::basic_istringstream<CharT> stream(STR("1.25"));
    stream.imbue(std::locale::classic());
    duration<int> result{};
    from_stream(stream, STR("%S").c_str(), result);
    assert(!stream.fail());
    assert(result == 1s);
    assert(stream.peek() == CharT('.'));
  }
  // Use the locale's comma decimal point and leave the trailing character unread.
  {
    std::basic_istringstream<CharT> stream(STR("1,25!"));
    stream.imbue(std::locale(std::locale::classic(), new comma_numpunct<CharT>));
    milliseconds result{};
    from_stream(stream, STR("%S").c_str(), result);
    assert(!stream.fail());
    assert(result == 1250ms);
    assert(stream.peek() == CharT('!'));
  }
}

// Test field widths, including signs, decimal points, and unread input.
template <class CharT>
void test_field_widths() {
  // A width on %F applies only to %Y.
  check(STR("002026-07-20"), STR("%6F"), 2026y / July / 20);

  // A sign consumes one character of the field's width.
  for (bool negative : {false, true}) {
    std::basic_istringstream<CharT> stream(negative ? STR("-123") : STR("+123"));
    stream.imbue(std::locale::classic());
    year result{};
    from_stream(stream, STR("%3Y").c_str(), result);
    assert(!stream.fail());
    assert(result == year{negative ? -12 : 12});
    assert(stream.peek() == CharT('3'));
  }

  // A width consumed entirely by the sign leaves no year digits.
  check_failure<year>(STR("+1"), STR("%1Y"));

  // Consume six characters and match the trailing literal without losing precision.
  check(STR("1.2345!"), STR("%6S!"), 1234500us);

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

// Test system, local, and file clock targets and their required date/time fields.
template <class CharT, class FileClock = file_clock>
void test_time_points() {
  const sys_seconds expected = sys_days{2026y / July / 20} + 13h + 45min + 30s;
  auto test = [&](auto value) { check(STR("20 26-07-20 01:45:30 PM"), STR("%C %y-%m-%d %I:%M:%S %p"), value); };
  // Combine century, year, and twelve-hour time into a system-clock time point.
  test(expected);
  // Parse the same calendar and time fields into a local time point.
  test(local_seconds{expected.time_since_epoch()});
  // Convert the parsed date and time to the file clock's epoch.
  if constexpr (requires { FileClock::from_sys(expected); }) {
    test(FileClock::from_sys(expected));
  }
#if !defined(TEST_HAS_NO_EXPERIMENTAL_TZDB) && !defined(TEST_HAS_NO_TIME_ZONE_DATABASE) &&                             \
    !defined(TEST_HAS_NO_FILESYSTEM)
  else {
    // Implementations may provide UTC conversions instead of system-clock conversions.
    test(FileClock::from_utc(utc_clock::from_sys(expected)));
  }

  // The GPS epoch is 1980-01-06 00:00:00.
  check(STR("1980-01-06 00:00:00"), STR("%F %T"), gps_seconds{0s});
#endif

  // A time of day alone cannot determine a time point.
  check_failure<sys_seconds>(STR("13:45:30"), STR("%H:%M:%S"));
  // Reject a nonnumeric day and preserve the original time point.
  check_failure<sys_seconds>(STR("2026-07-xx"), STR("%Y-%m-%d"), expected);

  // Parse a date into a system-clock time point with day precision.
  check(STR("2026-07-20"), STR("%F"), sys_days{2026y / July / 20});
  // A days target rejects time-of-day fields even when the UTC offset would cancel them.
  check_failure<sys_days>(STR("2026-07-20 04:00 +0400"), STR("%F %R %z"));
}

// Test time-point precision, representation, and conversion relative to the clock epoch.
template <class CharT>
void test_time_point_conversion() {
  // Preserve fractional seconds for a time point just before the Unix epoch.
  check(STR("1969-12-31 23:59:59.5"), STR("%F %T"), sys_time<milliseconds>{-500ms});
  // Preserve fractional seconds in a floating-point time point and match the trailing literal.
  check(STR("1970-01-01 00:00:12.125!"), STR("%F %T!"), sys_time<duration<double>>{duration<double>{12.125}});
  // Convert one second after the Unix epoch to an attosecond count.
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
// Test valid and invalid UTC leap seconds, including fractions and offsets.
template <class CharT>
void test_utc_leap_seconds() {
  const utc_seconds leap = utc_clock::from_sys(sys_days{2017y / January / 1}) - 1s;
  // Accept the leap second at the end of 2016.
  check(STR("2016-12-31 23:59:60"), STR("%F %T"), leap);
  // Apply the UTC offset and preserve the fraction within a leap second.
  check(STR("2017-01-01 00:59:60.5 +0100"), STR("%F %T %z"), leap + 500ms);
  // Reject second 60 at a time when no leap second occurs.
  check_failure<utc_seconds>(STR("2026-07-20 12:00:60"), STR("%F %T"));
}
#endif

// Test signed years, century selection, and agreement between year fields.
template <class CharT>
void test_year_resolution() {
  // Accept an explicit positive sign within the year field's width.
  check(STR("+123"), STR("%4Y"), year{123});
  // Accept a negative year with room for its sign and three digits.
  check(STR("-123"), STR("%4Y"), year{-123});
  // Year zero is a valid calendar year.
  check(STR("0"), STR("%Y"), year{0});

  // Two-digit years use a default century unless %C or %Y supplies one.
  check(STR("68"), STR("%y"), year{2068});
  // The default century changes from 2000 to 1900 at two-digit year 69.
  check(STR("69"), STR("%y"), year{1969});
  // Combine an explicit century with a two-digit year.
  check(STR("20 26"), STR("%C %y"), 2026y);
  // Combine a negative century with a two-digit year.
  check(STR("-20 76"), STR("%3C %y"), year{-1976});
  // Accept a full year followed by its matching century.
  check(STR("2026 20"), STR("%Y %C"), 2026y);
  // A full year supplies the century for a matching two-digit year.
  check(STR("1926 26"), STR("%Y %y"), 1926y);
  // The year in %F supplies the century for a following two-digit year.
  check(STR("1926-07-20 26"), STR("%F %y"), 1926y / July / 20);

  // Increasing the width does not make 100 a valid two-digit year.
  check_failure<year>(STR("100"), STR("%3y"));
  // A century alone cannot determine a complete year.
  check_failure<year>(STR("20"), STR("%C"));
}

// Test duration components, their ranges, and rejection of calendar fields.
// Parse time-of-day and day-count components.
template <class CharT>
void test_duration() {
  // Parse a standalone seconds field into a duration.
  check(STR("30"), STR("%S"), 30s);

  // Interpret %j as a day count when parsing a duration.
  check(STR("2"), STR("%j"), days{2});
  // Add hours and minutes to a multi-day duration.
  check(STR("2 01:30"), STR("%j %R"), 48h + 90min);
  // Accept the largest ordinary hour, minute, and second fields.
  check(STR("23:59:59"), STR("%T"), 23h + 59min + 59s);

  // Reject a calendar day-of-month field and preserve the original duration.
  check_failure<seconds>(STR("15 01"), STR("%d %H"), 42s);
}

// Test twelve-hour and twenty-four-hour fields, AM/PM, and hour ranges.
template <class CharT>
void test_hour_fields() {
  // Twelve o'clock with AM denotes midnight without a twenty-four-hour field.
  check(STR("12 AM"), STR("%I %p"), 0h);
  // Apply a PM designator that precedes the twelve-hour time.
  check(STR("PM 01:30"), STR("%p %I:%M"), 810min);

  // Accept matching twenty-four-hour, twelve-hour, and PM fields.
  check(STR("13 01 PM"), STR("%H %I %p"), 13h);
  // A preceding twenty-four-hour field resolves the twelve-hour field as PM.
  check(STR("13 01"), STR("%H %I"), 13h);
  // A following twenty-four-hour field also resolves the twelve-hour field as PM.
  check(STR("01 13"), STR("%I %H"), 13h);
  // Hour 00 resolves twelve o'clock as midnight.
  check(STR("00 12"), STR("%H %I"), 0h);
  // Hour 12 resolves twelve o'clock as noon.
  check(STR("12 12"), STR("%H %I"), 12h);

  // An AM/PM designator alone cannot determine a duration.
  check_failure<minutes>(STR("PM"), STR("%p"));
  // Hour 13 is outside the twelve-hour range even with a PM designator.
  check_failure<minutes>(STR("13 PM"), STR("%I %p"));
}

// Test duration representations and whether the target period can represent the parsed value.
template <class CharT>
void test_duration_conversion() {
  // Store a positive seconds count in an unsigned duration representation.
  check(STR("42"), STR("%S"), duration<unsigned>{42});
  // Scale fractional seconds to a floating-point millisecond count.
  check(STR("1.25"), STR("%S"), duration<double, std::milli>{1250});
  // An integral hours target cannot represent the additional thirty minutes.
  check_failure<hours>(STR("01:30"), STR("%R"));
}

// Test UTC offset syntax and its effect on the parsed time point.
template <class CharT>
void test_offsets() {
  // Consume the offset without storing it when the output pointer is null.
  check(STR("15 +0130"), STR("%d %z"), day{15});

  const sys_seconds date = sys_days{2026y / July / 20};
  // An unsigned two-digit offset supplies positive hours without minutes.
  check(STR("2026-07-20 00:00:00 04"), STR("%F %T %z"), date - 4h);
  // A four-digit offset supplies positive hours and minutes.
  check(STR("2026-07-20 00:00:00 0430"), STR("%F %T %z"), date - 4h - 30min);
  // Subtracting a negative UTC offset moves the result forward.
  check(STR("2026-07-20 00:00:00 -0130"), STR("%F %T %z"), date + 90min);

  // %Ez accepts a single-digit hour without minutes.
  check(STR("2026-07-20 00:00:00 4"), STR("%F %T %Ez"), date - 4h);
  // %Ez accepts an explicit positive sign and colon-separated minutes.
  check(STR("2026-07-20 00:00:00 +4:30"), STR("%F %T %Ez"), date - 4h - 30min);
  // %Oz accepts the same colon-separated offset syntax as %Ez.
  check(STR("2026-07-20 00:00:00 +4:30"), STR("%F %T %Oz"), date - 4h - 30min);
  // %Ez also accepts a negative colon-separated offset.
  check(STR("2026-07-20 00:00:00 -4:30"), STR("%F %T %Ez"), date + 4h + 30min);

  // A parsed UTC offset does not shift a local time point.
  check(STR("2026-07-20 00:00:00 +0130"), STR("%F %T %z"), local_seconds{date.time_since_epoch()});

  // Basic hours and any supplied minutes require two digits.
  check_failure<sys_seconds>(STR("2026-07-20 00:00:00 019!"), STR("%F %T %z!"));
  // Modified offsets still require two minute digits after the colon.
  check_failure<sys_seconds>(STR("2026-07-20 00:00:00 4:9!"), STR("%F %T %Ez!"));
  // An unmodified offset requires two hour digits.
  check_failure<sys_seconds>(STR("2026-07-20 00:00:00 4"), STR("%F %T %z"));
}

// Test consuming and storing time-zone abbreviations.
template <class CharT>
void test_abbrev() {
  // Consume the abbreviation without storing it when the output pointer is null.
  check(STR("15 UTC"), STR("%d %Z"), day{15});

  // Store letters, digits, and permitted punctuation in the abbreviation, stopping before '!'.
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
  test_compound_specifiers<CharT>();
  test_locale_formats<CharT>();
  test_literals_and_whitespace<CharT>();
  test_date_resolution<CharT>();
  test_modifier_E<CharT>();
  test_modifier_O<CharT>();
  test_fractional_seconds<CharT>();
  test_field_widths<CharT>();
  test_year_resolution<CharT>();
  test_duration<CharT>();
  test_hour_fields<CharT>();
  test_duration_conversion<CharT>();
  test_time_points<CharT>();
  test_time_point_conversion<CharT>();
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
