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

// template<class Duration, class TimeZonePtr = const time_zone*>
// class zoned_time;
//
// zoned_time& operator=(const zoned_time&) = default;

#include <cassert>
#include <chrono>
#include <concepts>

int main(int, char**) {
  std::chrono::zoned_time<std::chrono::seconds> zt{std::chrono::sys_seconds{std::chrono::seconds{42}}};
  assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
  assert(zt.get_sys_time() == std::chrono::sys_seconds{std::chrono::seconds{42}});

  std::chrono::zoned_time<std::chrono::seconds> copy;
  copy                                                                               = zt;
  std::same_as<std::chrono::zoned_time<std::chrono::seconds>&> decltype(auto) result = (copy = zt);
  assert(&result == &copy);
  assert(copy.get_time_zone() == std::chrono::locate_zone("UTC"));
  assert(copy.get_sys_time() == std::chrono::sys_seconds{std::chrono::seconds{42}});

  return 0;
}
