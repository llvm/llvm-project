//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

// const time_zone* current_zone();

#include <cassert>
#include <chrono>
#include <string_view>
#include <stdlib.h>

#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"

#ifdef _WIN32
static void set_tz(std::string zone) {
  // Note Windows does not have setenv, only putenv
  // https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/putenv-s-wputenv-s?view=msvc-170
  // Unlike POSIX it does not mention the string of putenv becomes part
  // of the environment.

  int status = _putenv_s("TZ", zone.c_str());
  assert(status == 0);
}

#else
static void set_tz(const std::string& zone) {
  int status = setenv("TZ", zone.c_str(), 1);
  assert(status == 0);
}
#endif

static void test_zone(const std::string& zone) {
  set_tz(zone);
  const std::chrono::time_zone* tz = std::chrono::current_zone();
  assert(tz);
  assert(tz->name() == zone);
}

static void test_link(const std::string& link, std::string_view zone) {
  set_tz(link);
  const std::chrono::time_zone* tz = std::chrono::current_zone();
  assert(tz);
  assert(tz->name() == zone);
}

int main(int, const char**) {
  const std::chrono::time_zone* tz = std::chrono::current_zone();
  // Returns a valid time zone, the value depends on the OS settings.
  assert(tz);
  // setting the environment to an invalid value returns the value of
  // the OS setting.
  set_tz("This is not a time zone");
  assert(tz == std::chrono::current_zone());

  const std::chrono::tzdb& db = std::chrono::get_tzdb();
  for (const auto& zone : db.zones)
    test_zone(std::string{zone.name()});

  for (const auto& link : db.links)
    test_link(std::string{link.name()}, link.target());

  return 0;
}
