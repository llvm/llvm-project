//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// For information see https://libcxx.llvm.org/DesignDocs/TimeZone.html

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "include/tzdb/time_zone_link_private.h"
#include "include/tzdb/time_zone_private.h"
#include "include/tzdb/types_private.h"
#include "include/tzdb/tzdb_list_private.h"
#include "include/tzdb/tzdb_private.h"

// Contains a parser for the IANA time zone data files.
//
// These files can be found at https://data.iana.org/time-zones/ and are in the
// public domain. Information regarding the input can be found at
// https://data.iana.org/time-zones/tz-how-to.html and
// https://man7.org/linux/man-pages/man8/zic.8.html.
//
// As indicated at https://howardhinnant.github.io/date/tz.html#Installation
// For Windows another file seems to be required
// https://raw.githubusercontent.com/unicode-org/cldr/master/common/supplemental/windowsZones.xml
// This file seems to contain the mapping of Windows time zone name to IANA
// time zone names.
//
// However this article mentions another way to do the mapping on Windows
// https://devblogs.microsoft.com/oldnewthing/20210527-00/?p=105255
// This requires Windows 10 Version 1903, which was released in May of 2019
// and considered end of life in December 2020
// https://learn.microsoft.com/en-us/lifecycle/announcements/windows-10-1903-end-of-servicing
//
// TODO TZDB Implement the Windows mapping in tzdb::current_zone

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono {

// This function is weak so it can be overriden in the tests. The
// declaration is in the test header test/support/test_tzdb.h
_LIBCPP_WEAK string_view __libcpp_tzdb_directory() {
#if defined(__linux__)
  return "/usr/share/zoneinfo/";
#else
#  error "unknown path to the IANA Time Zone Database"
#endif
}

//===----------------------------------------------------------------------===//
//                           Details
//===----------------------------------------------------------------------===//

[[nodiscard]] static bool __is_whitespace(int __c) { return __c == ' ' || __c == '\t'; }

static void __skip_optional_whitespace(istream& __input) {
  while (chrono::__is_whitespace(__input.peek()))
    __input.get();
}

static void __skip_mandatory_whitespace(istream& __input) {
  if (!chrono::__is_whitespace(__input.get()))
    std::__throw_runtime_error("corrupt tzdb: expected whitespace");

  chrono::__skip_optional_whitespace(__input);
}

[[nodiscard]] static bool __is_eol(int __c) { return __c == '\n' || __c == std::char_traits<char>::eof(); }

static void __skip_line(istream& __input) {
  while (!chrono::__is_eol(__input.peek())) {
    __input.get();
  }
  __input.get();
}

static void __skip(istream& __input, char __suffix) {
  if (std::tolower(__input.peek()) == __suffix)
    __input.get();
}

static void __skip(istream& __input, string_view __suffix) {
  for (auto __c : __suffix)
    if (std::tolower(__input.peek()) == __c)
      __input.get();
}

static void __matches(istream& __input, char __expected) {
  if (std::tolower(__input.get()) != __expected)
    std::__throw_runtime_error((string("corrupt tzdb: expected character '") + __expected + '\'').c_str());
}

static void __matches(istream& __input, string_view __expected) {
  for (auto __c : __expected)
    if (std::tolower(__input.get()) != __c)
      std::__throw_runtime_error((string("corrupt tzdb: expected string '") + string(__expected) + '\'').c_str());
}

[[nodiscard]] static string __parse_string(istream& __input) {
  string __result;
  while (true) {
    int __c = __input.get();
    switch (__c) {
    case ' ':
    case '\t':
    case '\n':
      __input.unget();
      [[fallthrough]];
    case istream::traits_type::eof():
      if (__result.empty())
        std::__throw_runtime_error("corrupt tzdb: expected a string");

      return __result;

    default:
      __result.push_back(__c);
    }
  }
}

[[nodiscard]] static int64_t __parse_integral(istream& __input, bool __leading_zero_allowed) {
  int64_t __result = __input.get();
  if (__leading_zero_allowed) {
    if (__result < '0' || __result > '9')
      std::__throw_runtime_error("corrupt tzdb: expected a digit");
  } else {
    if (__result < '1' || __result > '9')
      std::__throw_runtime_error("corrupt tzdb: expected a non-zero digit");
  }
  __result -= '0';
  while (true) {
    if (__input.peek() < '0' || __input.peek() > '9')
      return __result;

    // In order to avoid possible overflows we limit the accepted range.
    // Most values parsed are expected to be very small:
    // - 8784 hours in a year
    // - 31 days in a month
    // - year no real maximum, these values are expected to be less than
    //   the range of the year type.
    //
    // However the leapseconds use a seconds after epoch value. Using an
    // int would run into an overflow in 2038. By using a 64-bit value
    // the range is large enough for the bilions of years. Limiting that
    // range slightly to make the code easier is not an issue.
    if (__result > (std::numeric_limits<int64_t>::max() / 16))
      std::__throw_runtime_error("corrupt tzdb: integral too large");

    __result *= 10;
    __result += __input.get() - '0';
  }
}

//===----------------------------------------------------------------------===//
//                          Calendar
//===----------------------------------------------------------------------===//

[[nodiscard]] static day __parse_day(istream& __input) {
  unsigned __result = chrono::__parse_integral(__input, false);
  if (__result > 31)
    std::__throw_runtime_error("corrupt tzdb day: value too large");
  return day{__result};
}

[[nodiscard]] static weekday __parse_weekday(istream& __input) {
  // TZDB allows the shortest unique name.
  switch (std::tolower(__input.get())) {
  case 'f':
    chrono::__skip(__input, "riday");
    return Friday;

  case 'm':
    chrono::__skip(__input, "onday");
    return Monday;

  case 's':
    switch (std::tolower(__input.get())) {
    case 'a':
      chrono::__skip(__input, "turday");
      return Saturday;

    case 'u':
      chrono::__skip(__input, "nday");
      return Sunday;
    }
    break;

  case 't':
    switch (std::tolower(__input.get())) {
    case 'h':
      chrono::__skip(__input, "ursday");
      return Thursday;

    case 'u':
      chrono::__skip(__input, "esday");
      return Tuesday;
    }
    break;
  case 'w':
    chrono::__skip(__input, "ednesday");
    return Wednesday;
  }

  std::__throw_runtime_error("corrupt tzdb weekday: invalid name");
}

[[nodiscard]] static month __parse_month(istream& __input) {
  // TZDB allows the shortest unique name.
  switch (std::tolower(__input.get())) {
  case 'a':
    switch (std::tolower(__input.get())) {
    case 'p':
      chrono::__skip(__input, "ril");
      return April;

    case 'u':
      chrono::__skip(__input, "gust");
      return August;
    }
    break;

  case 'd':
    chrono::__skip(__input, "ecember");
    return December;

  case 'f':
    chrono::__skip(__input, "ebruary");
    return February;

  case 'j':
    switch (std::tolower(__input.get())) {
    case 'a':
      chrono::__skip(__input, "nuary");
      return January;

    case 'u':
      switch (std::tolower(__input.get())) {
      case 'n':
        chrono::__skip(__input, 'e');
        return June;

      case 'l':
        chrono::__skip(__input, 'y');
        return July;
      }
    }
    break;

  case 'm':
    if (std::tolower(__input.get()) == 'a')
      switch (std::tolower(__input.get())) {
      case 'y':
        return May;

      case 'r':
        chrono::__skip(__input, "ch");
        return March;
      }
    break;

  case 'n':
    chrono::__skip(__input, "ovember");
    return November;

  case 'o':
    chrono::__skip(__input, "ctober");
    return October;

  case 's':
    chrono::__skip(__input, "eptember");
    return September;
  }
  std::__throw_runtime_error("corrupt tzdb month: invalid name");
}

[[nodiscard]] static year __parse_year_value(istream& __input) {
  bool __negative = __input.peek() == '-';
  if (__negative) [[unlikely]]
    __input.get();

  int64_t __result = __parse_integral(__input, true);
  if (__result > static_cast<int>(year::max())) {
    if (__negative)
      std::__throw_runtime_error("corrupt tzdb year: year is less than the minimum");

    std::__throw_runtime_error("corrupt tzdb year: year is greater than the maximum");
  }

  return year{static_cast<int>(__negative ? -__result : __result)};
}

[[nodiscard]] static year __parse_year(istream& __input) {
  if (std::tolower(__input.peek()) != 'm') [[likely]]
    return chrono::__parse_year_value(__input);

  __input.get();
  switch (std::tolower(__input.peek())) {
  case 'i':
    __input.get();
    chrono::__skip(__input, 'n');
    [[fallthrough]];

  case ' ':
    // The m is minimum, even when that is ambiguous.
    return year::min();

  case 'a':
    __input.get();
    chrono::__skip(__input, 'x');
    return year::max();
  }

  std::__throw_runtime_error("corrupt tzdb year: expected 'min' or 'max'");
}

//===----------------------------------------------------------------------===//
//                        TZDB fields
//===----------------------------------------------------------------------===//

[[nodiscard]] static year __parse_to(istream& __input, year __only) {
  if (std::tolower(__input.peek()) != 'o')
    return chrono::__parse_year(__input);

  __input.get();
  chrono::__skip(__input, "nly");
  return __only;
}

[[nodiscard]] static __tz::__constrained_weekday::__comparison_t __parse_comparison(istream& __input) {
  switch (__input.get()) {
  case '>':
    chrono::__matches(__input, '=');
    return __tz::__constrained_weekday::__ge;

  case '<':
    chrono::__matches(__input, '=');
    return __tz::__constrained_weekday::__le;
  }
  std::__throw_runtime_error("corrupt tzdb on: expected '>=' or '<='");
}

[[nodiscard]] static __tz::__on __parse_on(istream& __input) {
  if (std::isdigit(__input.peek()))
    return chrono::__parse_day(__input);

  if (std::tolower(__input.peek()) == 'l') {
    chrono::__matches(__input, "last");
    return weekday_last(chrono::__parse_weekday(__input));
  }

  return __tz::__constrained_weekday{
      chrono::__parse_weekday(__input), chrono::__parse_comparison(__input), chrono::__parse_day(__input)};
}

[[nodiscard]] static seconds __parse_duration(istream& __input) {
  seconds __result{0};
  int __c         = __input.peek();
  bool __negative = __c == '-';
  if (__negative) {
    __input.get();
    // Negative is either a negative value or a single -.
    // The latter means 0 and the parsing is complete.
    if (!std::isdigit(__input.peek()))
      return __result;
  }

  __result += hours(__parse_integral(__input, true));
  if (__input.peek() != ':')
    return __negative ? -__result : __result;

  __input.get();
  __result += minutes(__parse_integral(__input, true));
  if (__input.peek() != ':')
    return __negative ? -__result : __result;

  __input.get();
  __result += seconds(__parse_integral(__input, true));
  if (__input.peek() != '.')
    return __negative ? -__result : __result;

  __input.get();
  (void)__parse_integral(__input, true); // Truncate the digits.

  return __negative ? -__result : __result;
}

[[nodiscard]] static __tz::__clock __parse_clock(istream& __input) {
  switch (__input.get()) { // case sensitive
  case 'w':
    return __tz::__clock::__local;
  case 's':
    return __tz::__clock::__standard;

  case 'u':
  case 'g':
  case 'z':
    return __tz::__clock::__universal;
  }

  __input.unget();
  return __tz::__clock::__local;
}

[[nodiscard]] static bool __parse_dst(istream& __input, seconds __offset) {
  switch (__input.get()) { // case sensitive
  case 's':
    return false;

  case 'd':
    return true;
  }

  __input.unget();
  return __offset != 0s;
}

[[nodiscard]] static __tz::__at __parse_at(istream& __input) {
  return {__parse_duration(__input), __parse_clock(__input)};
}

[[nodiscard]] static __tz::__save __parse_save(istream& __input) {
  seconds __time = chrono::__parse_duration(__input);
  return {__time, chrono::__parse_dst(__input, __time)};
}

[[nodiscard]] static string __parse_letters(istream& __input) {
  string __result = __parse_string(__input);
  // Canonicalize "-" to "" since they are equivalent in the specification.
  return __result != "-" ? __result : "";
}

[[nodiscard]] static __tz::__continuation::__rules_t __parse_rules(istream& __input) {
  int __c = __input.peek();
  // A single -  is not a SAVE but a special case.
  if (__c == '-') {
    __input.get();
    if (chrono::__is_whitespace(__input.peek()))
      return monostate{};
    __input.unget();
    return chrono::__parse_save(__input);
  }

  if (std::isdigit(__c) || __c == '+')
    return chrono::__parse_save(__input);

  return chrono::__parse_string(__input);
}

[[nodiscard]] static __tz::__continuation __parse_continuation(__tz::__rules_storage_type& __rules, istream& __input) {
  __tz::__continuation __result;

  __result.__rule_database_ = std::addressof(__rules);

  // Note STDOFF is specified as
  //   This field has the same format as the AT and SAVE fields of rule lines;
  // These fields have different suffix letters, these letters seem
  // not to be used so do not allow any of them.

  __result.__stdoff = chrono::__parse_duration(__input);
  chrono::__skip_mandatory_whitespace(__input);
  __result.__rules = chrono::__parse_rules(__input);
  chrono::__skip_mandatory_whitespace(__input);
  __result.__format = chrono::__parse_string(__input);
  chrono::__skip_optional_whitespace(__input);

  if (chrono::__is_eol(__input.peek()))
    return __result;
  __result.__year = chrono::__parse_year(__input);
  chrono::__skip_optional_whitespace(__input);

  if (chrono::__is_eol(__input.peek()))
    return __result;
  __result.__in = chrono::__parse_month(__input);
  chrono::__skip_optional_whitespace(__input);

  if (chrono::__is_eol(__input.peek()))
    return __result;
  __result.__on = chrono::__parse_on(__input);
  chrono::__skip_optional_whitespace(__input);

  if (chrono::__is_eol(__input.peek()))
    return __result;
  __result.__at = __parse_at(__input);

  return __result;
}

//===----------------------------------------------------------------------===//
//                   Time Zone Database entries
//===----------------------------------------------------------------------===//

static string __parse_version(istream& __input) {
  // The first line in tzdata.zi contains
  //    # version YYYYw
  // The parser expects this pattern
  // #\s*version\s*\(.*)
  // This part is not documented.
  chrono::__matches(__input, '#');
  chrono::__skip_optional_whitespace(__input);
  chrono::__matches(__input, "version");
  chrono::__skip_mandatory_whitespace(__input);
  return chrono::__parse_string(__input);
}

static void __parse_rule(tzdb& __tzdb, __tz::__rules_storage_type& __rules, istream& __input) {
  chrono::__skip_mandatory_whitespace(__input);
  string __name = chrono::__parse_string(__input);

  if (__rules.empty() || __rules.back().first != __name)
    __rules.emplace_back(__name, vector<__tz::__rule>{});

  __tz::__rule& __rule = __rules.back().second.emplace_back();

  chrono::__skip_mandatory_whitespace(__input);
  __rule.__from = chrono::__parse_year(__input);
  chrono::__skip_mandatory_whitespace(__input);
  __rule.__to = chrono::__parse_to(__input, __rule.__from);
  chrono::__skip_mandatory_whitespace(__input);
  chrono::__matches(__input, '-');
  chrono::__skip_mandatory_whitespace(__input);
  __rule.__in = chrono::__parse_month(__input);
  chrono::__skip_mandatory_whitespace(__input);
  __rule.__on = chrono::__parse_on(__input);
  chrono::__skip_mandatory_whitespace(__input);
  __rule.__at = __parse_at(__input);
  chrono::__skip_mandatory_whitespace(__input);
  __rule.__save = __parse_save(__input);
  chrono::__skip_mandatory_whitespace(__input);
  __rule.__letters = chrono::__parse_letters(__input);
  chrono::__skip_line(__input);
}

static void __parse_zone(tzdb& __tzdb, __tz::__rules_storage_type& __rules, istream& __input) {
  chrono::__skip_mandatory_whitespace(__input);
  auto __p                                      = std::make_unique<time_zone::__impl>(chrono::__parse_string(__input));
  vector<__tz::__continuation>& __continuations = __p->__continuations();
  chrono::__skip_mandatory_whitespace(__input);

  do {
    // The first line must be valid, continuations are optional.
    __continuations.emplace_back(__parse_continuation(__rules, __input));
    chrono::__skip_line(__input);
    chrono::__skip_optional_whitespace(__input);
  } while (std::isdigit(__input.peek()) || __input.peek() == '-');

  __tzdb.zones.emplace_back(time_zone::__create(std::move(__p)));
}

static void __parse_link(tzdb& __tzdb, istream& __input) {
  chrono::__skip_mandatory_whitespace(__input);
  string __target = chrono::__parse_string(__input);
  chrono::__skip_mandatory_whitespace(__input);
  string __name = chrono::__parse_string(__input);
  chrono::__skip_line(__input);

  __tzdb.links.emplace_back(time_zone_link::__constructor_tag{}, std::move(__name), std::move(__target));
}

static void __parse_tzdata(tzdb& __db, __tz::__rules_storage_type& __rules, istream& __input) {
  while (true) {
    int __c = std::tolower(__input.get());

    switch (__c) {
    case istream::traits_type::eof():
      return;

    case ' ':
    case '\t':
    case '\n':
      break;

    case '#':
      chrono::__skip_line(__input);
      break;

    case 'r':
      chrono::__skip(__input, "ule");
      chrono::__parse_rule(__db, __rules, __input);
      break;

    case 'z':
      chrono::__skip(__input, "one");
      chrono::__parse_zone(__db, __rules, __input);
      break;

    case 'l':
      chrono::__skip(__input, "ink");
      chrono::__parse_link(__db, __input);
      break;

    default:
      std::__throw_runtime_error("corrupt tzdb: unexpected input");
    }
  }
}

void __init_tzdb(tzdb& __tzdb, __tz::__rules_storage_type& __rules) {
  filesystem::path __root = chrono::__libcpp_tzdb_directory();
  ifstream __tzdata{__root / "tzdata.zi"};

  __tzdb.version = chrono::__parse_version(__tzdata);
  chrono::__parse_tzdata(__tzdb, __rules, __tzdata);
  std::ranges::sort(__tzdb.zones);
  std::ranges::sort(__tzdb.links);
  std::ranges::sort(__rules, {}, [](const auto& p) { return p.first; });
}

//===----------------------------------------------------------------------===//
//                           Public API
//===----------------------------------------------------------------------===//

_LIBCPP_NODISCARD_EXT _LIBCPP_AVAILABILITY_TZDB _LIBCPP_EXPORTED_FROM_ABI tzdb_list& get_tzdb_list() {
  static tzdb_list __result{new tzdb_list::__impl()};
  return __result;
}

_LIBCPP_AVAILABILITY_TZDB _LIBCPP_EXPORTED_FROM_ABI const tzdb& reload_tzdb() {
  if (chrono::remote_version() == chrono::get_tzdb().version)
    return chrono::get_tzdb();

  return chrono::get_tzdb_list().__implementation().__load();
}

_LIBCPP_NODISCARD_EXT _LIBCPP_AVAILABILITY_TZDB _LIBCPP_EXPORTED_FROM_ABI string remote_version() {
  filesystem::path __root = chrono::__libcpp_tzdb_directory();
  ifstream __tzdata{__root / "tzdata.zi"};
  return chrono::__parse_version(__tzdata);
}

} // namespace chrono

_LIBCPP_END_NAMESPACE_STD
