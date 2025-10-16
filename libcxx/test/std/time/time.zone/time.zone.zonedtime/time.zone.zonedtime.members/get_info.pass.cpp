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

namespace cr = std::chrono;

int main(int, char**) {
  {
    cr::zoned_time<cr::seconds> zt;

    std::same_as<cr::sys_info> decltype(auto) info = zt.get_info();
    assert(info.begin == cr::sys_seconds::min());
    assert(info.end == cr::sys_seconds::max());
    assert(info.offset == cr::seconds{0});
    assert(info.save == cr::minutes{0});
    assert(info.abbrev == "UTC");
  }
  {
    const cr::zoned_time<cr::seconds> zt;

    std::same_as<cr::sys_info> decltype(auto) info = zt.get_info();
    assert(info.begin == cr::sys_seconds::min());
    assert(info.end == cr::sys_seconds::max());
    assert(info.offset == cr::seconds{0});
    assert(info.save == cr::minutes{0});
    assert(info.abbrev == "UTC");
  }

  return 0;
}
