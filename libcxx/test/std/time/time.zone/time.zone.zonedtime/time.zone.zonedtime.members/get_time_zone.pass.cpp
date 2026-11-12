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
// TimeZonePtr get_time_zone() const;

#include <chrono>
#include <cassert>
#include <concepts>

#include "../test_offset_time_zone.h"

int main(int, char**) {
  {
    const std::chrono::time_zone* tz = std::chrono::locate_zone("UTC");
    std::chrono::zoned_time<std::chrono::seconds> zt{tz};

    std::same_as<const std::chrono::time_zone*> decltype(auto) ptr = zt.get_time_zone();
    assert(ptr = tz);
  }

  {
    int tz = 0;
    std::chrono::zoned_time<std::chrono::seconds, int*> zt{&tz};

    std::same_as<int*> decltype(auto) ptr = zt.get_time_zone();
    assert(ptr = &tz);
  }

  {
    int tz = 0;
    const std::chrono::zoned_time<std::chrono::seconds, int*> zt{&tz};

    std::same_as<int*> decltype(auto) ptr = zt.get_time_zone();
    assert(ptr = &tz);
  }

  return 0;
}
