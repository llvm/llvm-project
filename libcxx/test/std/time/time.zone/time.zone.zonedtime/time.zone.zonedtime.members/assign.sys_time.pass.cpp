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
// zoned_time& operator=(const sys_time<Duration>& st);

#include <cassert>
#include <chrono>
#include <concepts>
#include <type_traits>

int main(int, char**) {
  {
    using duration   = std::chrono::nanoseconds;
    using time_point = std::chrono::sys_time<duration>;
    using zoned_time = std::chrono::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }
  {
    using duration   = std::chrono::microseconds;
    using time_point = std::chrono::sys_time<duration>;
    using zoned_time = std::chrono::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }
  {
    using duration   = std::chrono::milliseconds;
    using time_point = std::chrono::sys_time<duration>;
    using zoned_time = std::chrono::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }
  {
    using duration   = std::chrono::seconds;
    using time_point = std::chrono::sys_time<duration>;
    using zoned_time = std::chrono::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }
  {
    using duration   = std::chrono::days;
    using time_point = std::chrono::sys_time<duration>;
    using zoned_time = std::chrono::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }
  {
    using duration   = std::chrono::weeks;
    using time_point = std::chrono::sys_time<duration>;
    using zoned_time = std::chrono::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }
  {
    using duration   = std::chrono::months;
    using time_point = std::chrono::sys_time<duration>;
    using zoned_time = std::chrono::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }
  {
    using duration   = std::chrono::years;
    using time_point = std::chrono::sys_time<duration>;
    using zoned_time = std::chrono::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) _ = zt = time_point{duration{99}};
    assert(zt.get_time_zone() == std::chrono::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }

  return 0;
}
