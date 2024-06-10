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
// sys_info get_info() const;

#include <cassert>
#include <chrono>
#include <concepts>

int main(int, char**) {
  {
    std::chrono::zoned_time<std::chrono::seconds> zt;

    std::same_as<std::chrono::sys_info> decltype(auto) info = zt.get_info();
    assert(info.begin == std::chrono::sys_seconds::min());
    assert(info.end == std::chrono::sys_seconds::max());
    assert(info.offset == std::chrono::seconds{0});
    assert(info.save == std::chrono::minutes{0});
    assert(info.abbrev == "UTC");
  }
  {
    const std::chrono::zoned_time<std::chrono::seconds> zt;

    std::same_as<std::chrono::sys_info> decltype(auto) info = zt.get_info();
    assert(info.begin == std::chrono::sys_seconds::min());
    assert(info.end == std::chrono::sys_seconds::max());
    assert(info.offset == std::chrono::seconds{0});
    assert(info.save == std::chrono::minutes{0});
    assert(info.abbrev == "UTC");
  }

  return 0;
}
