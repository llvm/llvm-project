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

//  struct tzdb {
//    string                 version;
//    vector<time_zone>      zones;
//    vector<time_zone_link> links;
//    vector<leap_second>    leap_seconds;
//
//    ...
//  };

#include <chrono>
#include <cassert>
#include <concepts>
#include <string>

#include "assert_macros.h"

int main(int, const char**) {
  std::chrono::tzdb tzdb;

  static_assert(std::same_as<decltype(tzdb.version), std::string>);
  tzdb.version = "version";
  assert(tzdb.version == "version");

  [[maybe_unused]] std::vector<std::chrono::time_zone>& zones          = tzdb.zones;
  [[maybe_unused]] std::vector<std::chrono::time_zone_link>& links = tzdb.links;
  [[maybe_unused]] std::vector<std::chrono::leap_second>& leap_seconds = tzdb.leap_seconds;

  return 0;
}
