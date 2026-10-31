//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

// template<>
// struct clock_time_conversion<utc_clock, system_clock> {
//   template<class Duration>
//     utc_time<common_type_t<Duration, seconds>>
//       operator()(const sys_time<Duration>& t) const;
// };
//
// template<>
// struct clock_time_conversion<system_clock, utc_clock> {
//   template<class Duration>
//     sys_time<common_type_t<Duration, seconds>>
//       operator()(const utc_time<Duration>& u) const;
// };

#include <cassert>
#include <chrono>
#include <concepts>
#include <type_traits>

template <class Duration>
void test_from_sys(Duration d) {
  const std::chrono::clock_time_conversion<std::chrono::utc_clock, std::chrono::system_clock> conversion{};
  const std::chrono::sys_time<Duration> t{d};

  std::same_as<std::chrono::utc_time<std::common_type_t<Duration, std::chrono::seconds>>> decltype(auto) result =
      conversion(t);
  assert(result == std::chrono::utc_clock::from_sys(t));
}

template <class Duration>
void test_to_sys(Duration d) {
  const std::chrono::clock_time_conversion<std::chrono::system_clock, std::chrono::utc_clock> conversion{};
  const std::chrono::utc_time<Duration> u{d};

  std::same_as<std::chrono::sys_time<std::common_type_t<Duration, std::chrono::seconds>>> decltype(auto) result =
      conversion(u);
  assert(result == std::chrono::utc_clock::to_sys(u));
}

template <class Duration>
void test(Duration d) {
  test_from_sys(d);
  test_to_sys(d);
}

int main(int, char**) {
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

  test(0s);
  test(-1'234'567'890ns);
  test(1'234'567'890'123'456'789ns);
  test(42ms);
  test(cr::minutes{-60});
  test(cr::days{20'000});
  test(cr::duration<double>{1.5e9});

  {
    // [time.clock.utc.overview]/1: utc_clock and system_clock share the same
    // epoch, and the leap seconds inserted since then are counted by utc_clock.
    const cr::clock_time_conversion<cr::utc_clock, cr::system_clock> to_utc{};
    const cr::clock_time_conversion<cr::system_clock, cr::utc_clock> to_sys{};

    assert(to_utc(cr::sys_seconds{0s}) == cr::utc_seconds{0s});
    assert(to_sys(cr::utc_seconds{0s}) == cr::sys_seconds{0s});

    // 27 positive leap seconds were inserted between 1972 and 2017.
    const cr::sys_seconds sys_2017 = cr::sys_days{cr::January / 1 / 2017};
    const cr::utc_seconds utc_2017 = to_utc(sys_2017);
    assert(utc_2017.time_since_epoch() == sys_2017.time_since_epoch() + 27s);
    assert(to_sys(utc_2017) == sys_2017);

    // During the leap second inserted at 2016-12-31 23:59:60, to_sys returns
    // the last representable value prior to the insertion.
    assert(to_sys(utc_2017 - 1s) == sys_2017 - 1s);
    assert(to_sys(utc_2017 - 500ms) == sys_2017 - 1ms);
    assert(to_sys(utc_2017 - 2s) == sys_2017 - 1s);
  }

  return 0;
}
