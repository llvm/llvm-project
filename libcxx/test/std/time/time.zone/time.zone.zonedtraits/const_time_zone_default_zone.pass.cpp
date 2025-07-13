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

// template<> struct zoned_traits<const time_zone*>;

// static const time_zone* default_zone();

#include <cassert>
#include <chrono>
#include <concepts>

int main(int, char**) {
  std::same_as<const std::chrono::time_zone*> decltype(auto) tz =
      std::chrono::zoned_traits<const std::chrono::time_zone*>::default_zone();
  assert(tz);

  // The time zone "UTC" can be a link, this means tz->name() can be something
  // differently. For example, "Etc/UTC". Instead validate whether same time
  // zone is returned by comparing the addresses.
  const std::chrono::time_zone* expected = std::chrono::locate_zone("UTC");
  assert(tz == expected);

  return 0;
}
