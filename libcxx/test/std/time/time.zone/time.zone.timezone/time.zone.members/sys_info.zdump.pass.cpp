//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb, has-no-zdump
// REQUIRES: long_tests

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

#include <chrono>
#include <format>
#include <fstream>
#include <cassert>

#include "filesystem_test_helper.h"
#include "assert_macros.h"
#include "concat_macros.h"

// The year range to validate. The dates used in practice are expected to be
// inside the tested range.
constexpr std::chrono::year first{1800};
constexpr std::chrono::year last{sizeof(time_t) == 8 ? 2100 : 2037};

// A custom sys_info class that also stores the name of the time zone.
// Its formatter matches the output of zdump.
struct sys_info : public std::chrono::sys_info {
  sys_info(std::string_view name_, std::chrono::sys_info info) : std::chrono::sys_info{info}, name{name_} {}

  std::string name;
};

template <>
struct std::formatter<sys_info, char> {
  template <class ParseContext>
  constexpr typename ParseContext::iterator parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <class FormatContext>
  typename FormatContext::iterator format(const sys_info& info, FormatContext& ctx) const {
    using namespace std::literals::chrono_literals;

    // Every "sys_info" entry of zdump consists of 2 lines.
    // - 1 for first second of the range
    // - 1 for last second of the range
    // For example:
    // Africa/Casablanca  Sun Mar 25 02:00:00 2018 UT = Sun Mar 25 03:00:00 2018 +01 isdst=1 gmtoff=3600
    // Africa/Casablanca  Sun May 13 01:59:59 2018 UT = Sun May 13 02:59:59 2018 +01 isdst=1 gmtoff=3600

    if (info.begin != std::chrono::sys_seconds::min())
      ctx.advance_to(std::format_to(
          ctx.out(),
          "{}  {:%a %b %e %H:%M:%S %Y} UT = {:%a %b %e %H:%M:%S %Y} {} isdst={:d} gmtoff={:%Q}\n",
          info.name,
          info.begin,
          info.begin + info.offset,
          info.abbrev,
          info.save != 0s,
          info.offset));

    if (info.end != std::chrono::sys_seconds::max())
      ctx.advance_to(std::format_to(
          ctx.out(),
          "{}  {:%a %b %e %H:%M:%S %Y} UT = {:%a %b %e %H:%M:%S %Y} {} isdst={:d} gmtoff={:%Q}\n",
          info.name,
          info.end - 1s,
          info.end - 1s + info.offset,
          info.abbrev,
          info.save != 0s,
          info.offset));

    return ctx.out();
  }
};

void process(std::ostream& stream, const std::chrono::time_zone& zone) {
  using namespace std::literals::chrono_literals;

  constexpr auto begin = std::chrono::time_point_cast<std::chrono::seconds>(
      static_cast<std::chrono::sys_days>(std::chrono::year_month_day{first, std::chrono::January, 1d}));
  constexpr auto end = std::chrono::time_point_cast<std::chrono::seconds>(
      static_cast<std::chrono::sys_days>(std::chrono::year_month_day{last, std::chrono::January, 1d}));

  std::chrono::sys_seconds s = begin;
  do {
    sys_info info{zone.name(), zone.get_info(s)};

    if (info.end >= end)
      info.end = std::chrono::sys_seconds::max();

    stream << std::format("{}", info);
    s = info.end;
  } while (s != std::chrono::sys_seconds::max());
}

// This test compares the output of the zdump against the output based on the
// standard library implementation. It tests all available time zones and
// validates them. The specification of how to use the IANA database is limited
// and the real database contains quite a number of "interesting" cases.
int main(int, const char**) {
  scoped_test_env env;
  const std::string file = env.create_file("zdump.txt");

  const std::chrono::tzdb& tzdb = std::chrono::get_tzdb();
  for (const auto& zone : tzdb.zones) {
    std::stringstream libcxx;
    process(libcxx, zone);

    int result = std::system(std::format("zdump -V -c{},{} {} > {}", first, last, zone.name(), file).c_str());
    assert(result == 0);

    std::stringstream zdump;
    zdump << std::ifstream(file).rdbuf();

    TEST_REQUIRE(
        libcxx.str() == zdump.str(),
        TEST_WRITE_CONCATENATED("\nTZ=", zone.name(), "\nlibc++\n", libcxx.str(), "|\n\nzdump\n", zdump.str(), "|"));
  }

  return 0;
}
