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
//
// class tai_clock;

// static utc_time<common_type_t<_Duration, seconds>>
// to_utc(const tai_time<_Duration>& __time) noexcept;

#include <chrono>
#include <cassert>
#include <source_location>

#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"

static void test_known_values() {
  namespace cr = std::chrono;
  using namespace std::literals::chrono_literals;
  constexpr auto unix_to_tai_epoch_offset = cr::sys_days{cr::January / 1 / 1970} - cr::sys_days{cr::January / 1 / 1958};

  // [time.clock.tai.overview]/1
  // ... 1958-01-01 00:00:00 TAI is equivalent to 1957-12-31 23:59:50 UTC
  // ... 2000-01-01 00:00:00 UTC is equivalent to 2000-01-01 00:00:32 TAI

  assert(cr::tai_clock::to_utc(cr::tai_seconds{0s}) ==
         cr::utc_clock::from_sys(cr::sys_days{cr::January / 1 / 1958} - 10s));

  assert(cr::tai_clock::to_utc(
             cr::tai_seconds{(cr::sys_days{cr::January / 1 / 2000} + unix_to_tai_epoch_offset).time_since_epoch()} +
             32s) == cr::utc_clock::from_sys(cr::sys_days{cr::January / 1 / 2000}));
}

template <class Duration>
static void test_leap_seconds(std::chrono::tai_time<Duration> tai,
                              std::chrono::utc_time<Duration> expected,
                              std::source_location loc = std::source_location::current()) {
  auto utc = std::chrono::tai_clock::to_utc(tai);
  TEST_REQUIRE(utc == expected,
               TEST_WRITE_CONCATENATED(loc, "\nExpected output ", expected, "\nActual output   ", utc, '\n'));
}

// Tests set if existing database entries at the time of writing.
static void test_transitions() {
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

  // "sys" is the time of the transition to the next leap second.
  // "elapsed" is the number of leap seconds before the transition.
  auto test_transition = [](cr::sys_days sys, cr::seconds elapsed) {
    constexpr auto unix_to_tai_epoch_offset =
        cr::sys_days{cr::January / 1 / 1970} - cr::sys_days{cr::January / 1 / 1958};
    cr::tai_seconds tai{sys.time_since_epoch() + unix_to_tai_epoch_offset + elapsed};

    test_leap_seconds(tai - 1ns, cr::utc_clock::from_sys(sys - 1ns));
    test_leap_seconds(tai + 1s, cr::utc_clock::from_sys(sys));
    test_leap_seconds(tai + 1s + 1ns, cr::utc_clock::from_sys(sys + 1ns));
  };

  // Transitions from the start of UTC.
  test_transition(cr::sys_days{cr::July / 1 / 1972}, 10s);
  test_transition(cr::sys_days{cr::January / 1 / 1973}, 11s);
  test_transition(cr::sys_days{cr::January / 1 / 1974}, 12s);
  test_transition(cr::sys_days{cr::January / 1 / 1975}, 13s);
  test_transition(cr::sys_days{cr::January / 1 / 1976}, 14s);
  test_transition(cr::sys_days{cr::January / 1 / 1977}, 15s);
  test_transition(cr::sys_days{cr::January / 1 / 1978}, 16s);
  test_transition(cr::sys_days{cr::January / 1 / 1979}, 17s);
  test_transition(cr::sys_days{cr::January / 1 / 1980}, 18s);
  test_transition(cr::sys_days{cr::July / 1 / 1981}, 19s);
  test_transition(cr::sys_days{cr::July / 1 / 1982}, 20s);
  test_transition(cr::sys_days{cr::July / 1 / 1983}, 21s);
  test_transition(cr::sys_days{cr::July / 1 / 1985}, 22s);
  test_transition(cr::sys_days{cr::January / 1 / 1988}, 23s);
  test_transition(cr::sys_days{cr::January / 1 / 1990}, 24s);
  test_transition(cr::sys_days{cr::January / 1 / 1991}, 25s);
  test_transition(cr::sys_days{cr::July / 1 / 1992}, 26s);
  test_transition(cr::sys_days{cr::July / 1 / 1993}, 27s);
  test_transition(cr::sys_days{cr::July / 1 / 1994}, 28s);
  test_transition(cr::sys_days{cr::January / 1 / 1996}, 29s);
  test_transition(cr::sys_days{cr::July / 1 / 1997}, 30s);
  test_transition(cr::sys_days{cr::January / 1 / 1999}, 31s);
  test_transition(cr::sys_days{cr::January / 1 / 2006}, 32s);
  test_transition(cr::sys_days{cr::January / 1 / 2009}, 33s);
  test_transition(cr::sys_days{cr::July / 1 / 2012}, 34s);
  test_transition(cr::sys_days{cr::July / 1 / 2015}, 35s);
  test_transition(cr::sys_days{cr::January / 1 / 2017}, 36s);
}

// Tests whether the return type is the expected type.
static void test_return_type() {
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::nanoseconds>> decltype(auto) _ =
        cr::tai_clock::to_utc(cr::tai_time<cr::nanoseconds>{0ns});
  }
  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::microseconds>> decltype(auto) _ =
        cr::tai_clock::to_utc(cr::tai_time<cr::microseconds>{0us});
  }
  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::milliseconds>> decltype(auto) _ =
        cr::tai_clock::to_utc(cr::tai_time<cr::milliseconds>{0ms});
  }

  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::seconds>> decltype(auto) _ =
        cr::tai_clock::to_utc(cr::tai_time<cr::seconds>{cr::seconds{0}});
  }

  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::seconds>> decltype(auto) _ =
        cr::tai_clock::to_utc(cr::tai_time<cr::minutes>{cr::minutes{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::seconds>> decltype(auto) _ =
        cr::tai_clock::to_utc(cr::tai_time<cr::hours>{cr::hours{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::seconds>> decltype(auto) _ =
        cr::tai_clock::to_utc(cr::tai_time<cr::days>{cr::days{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::seconds>> decltype(auto) _ =
        cr::tai_clock::to_utc(cr::tai_time<cr::weeks>{cr::weeks{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::seconds>> decltype(auto) _ =
        cr::tai_clock::to_utc(cr::tai_time<cr::months>{cr::months{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::seconds>> decltype(auto) _ =
        cr::tai_clock::to_utc(cr::tai_time<cr::years>{cr::years{0}});
  }
}

int main(int, const char**) {
  using namespace std::literals::chrono_literals;

  std::chrono::tai_seconds time = std::chrono::tai_seconds{0s};
  static_assert(noexcept(std::chrono::tai_clock::to_utc(time)));

  test_known_values();
  test_transitions();
  test_return_type();

  return 0;
}
