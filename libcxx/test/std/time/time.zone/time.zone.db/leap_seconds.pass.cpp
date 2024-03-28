//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-incomplete-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

// Tests the loaded leap seconds match
// https://eel.is/c++draft/time.zone.leap.overview#2
//
// At the moment of writing that list is the actual list.
// If in the future more leap seconds are added, the returned list may have more

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <ranges>

using namespace std::literals::chrono_literals;

// The list of leap seconds matching
// https://eel.is/c++draft/time.zone.leap.overview#2
// At the moment of writing that list is the actual list in the IANA database.
// If in the future more leap seconds can be added. Testng th
static const std::array /*<std::chrono::leap_second>*/ leap_seconds = {
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1972y / std::chrono::January / 1}}, 10s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1972y / std::chrono::July / 1}}, 11s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1973y / std::chrono::January / 1}}, 12s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1974y / std::chrono::January / 1}}, 13s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1975y / std::chrono::January / 1}}, 14s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1976y / std::chrono::January / 1}}, 15s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1977y / std::chrono::January / 1}}, 16s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1978y / std::chrono::January / 1}}, 17s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1979y / std::chrono::January / 1}}, 18s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1980y / std::chrono::January / 1}}, 19s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1981y / std::chrono::July / 1}}, 20s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1982y / std::chrono::July / 1}}, 21s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1983y / std::chrono::July / 1}}, 22s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1985y / std::chrono::July / 1}}, 23s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1988y / std::chrono::January / 1}}, 24s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1990y / std::chrono::January / 1}}, 25s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1991y / std::chrono::January / 1}}, 26s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1992y / std::chrono::July / 1}}, 27s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1993y / std::chrono::July / 1}}, 28s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1994y / std::chrono::July / 1}}, 29s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1996y / std::chrono::January / 1}}, 30s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1997y / std::chrono::July / 1}}, 31s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{1999y / std::chrono::January / 1}}, 32s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{2006y / std::chrono::January / 1}}, 33s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{2009y / std::chrono::January / 1}}, 34s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{2012y / std::chrono::July / 1}}, 35s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{2015y / std::chrono::July / 1}}, 36s),
    std::make_pair(std::chrono::sys_seconds{std::chrono::sys_days{2017y / std::chrono::January / 1}}, 37s)};

int main(int, const char**) {
  const std::chrono::tzdb& tzdb = std::chrono::get_tzdb();

  assert(tzdb.leap_seconds.size() >= leap_seconds.size());
  assert((std::ranges::equal(
      leap_seconds,
      tzdb.leap_seconds | std::ranges::views::take(leap_seconds.size()),
      [](const auto& lhs, const auto& rhs) { return lhs.first == rhs.date() && lhs.second == rhs.value(); })));

  return 0;
}
