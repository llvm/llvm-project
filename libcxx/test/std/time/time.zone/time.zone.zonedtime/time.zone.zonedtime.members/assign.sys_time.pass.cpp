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

namespace cr = std::chrono;

int main(int, char**) {
  {
    using duration   = cr::nanoseconds;
    using time_point = cr::sys_time<duration>;
    using zoned_time = cr::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) result = zt = time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }
  {
    using duration   = cr::microseconds;
    using time_point = cr::sys_time<duration>;
    using zoned_time = cr::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) result = zt = time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }
  {
    using duration   = cr::milliseconds;
    using time_point = cr::sys_time<duration>;
    using zoned_time = cr::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) result = zt = time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }
  {
    using duration   = cr::seconds;
    using time_point = cr::sys_time<duration>;
    using zoned_time = cr::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) result = zt = time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }
  {
    using duration   = cr::days;
    using time_point = cr::sys_time<duration>;
    using zoned_time = cr::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) result = zt = time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }
  {
    using duration   = cr::weeks;
    using time_point = cr::sys_time<duration>;
    using zoned_time = cr::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) result = zt = time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }
  {
    using duration   = cr::months;
    using time_point = cr::sys_time<duration>;
    using zoned_time = cr::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) result = zt = time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }
  {
    using duration   = cr::years;
    using time_point = cr::sys_time<duration>;
    using zoned_time = cr::zoned_time<duration>;
    zoned_time zt{time_point{duration{42}}};

    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{42}});

    std::same_as<zoned_time&> decltype(auto) result = zt = time_point{duration{99}};
    assert(&result == &zt);
    assert(zt.get_time_zone() == cr::locate_zone("UTC"));
    assert(zt.get_sys_time() == time_point{duration{99}});
  }

  return 0;
}
