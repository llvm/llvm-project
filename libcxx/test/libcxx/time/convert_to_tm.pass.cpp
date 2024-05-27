//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>

// template <class _Tm, class _ChronoT>
// _LIBCPP_HIDE_FROM_ABI _Tm __convert_to_tm(const _ChronoT& __value)

// Most of the code is tested indirectly in the chrono formatters. This only
// tests the hour overflow.

#include <__chrono/convert_to_tm.h>
#include <chrono>
#include <cassert>
#include <format>
#include <string_view>

#include "test_macros.h"

// libc++ uses a long as representation in std::chrono::hours.
// std::tm uses an int for its integral members. The overflow in the hour
// conversion can only occur on platforms where sizeof(long) > sizeof(int).
// Instead emulate this error by using a "tm" with shorts.
// (The function is already templated to this is quite easy to do,)
struct minimal_short_tm {
  short tm_sec;
  short tm_min;
  short tm_hour;
  const char* tm_zone;
};

int main(int, char**) {
  { // Test with the maximum number of hours that fit in a short.
    std::chrono::hh_mm_ss time{std::chrono::hours{32767}};
    minimal_short_tm result = std::__convert_to_tm<minimal_short_tm>(time);
    assert(result.tm_sec == 0);
    assert(result.tm_min == 0);
    assert(result.tm_hour == 32767);
  }

#ifndef TEST_HAS_NO_EXCEPTIONS
  { // Test above the maximum number of hours that fit in a short.
    std::chrono::hh_mm_ss time{std::chrono::hours{32768}};
    try {
      TEST_IGNORE_NODISCARD std::__convert_to_tm<minimal_short_tm>(time);
      assert(false);
    } catch ([[maybe_unused]] const std::format_error& e) {
      LIBCPP_ASSERT(e.what() == std::string_view("Formatting hh_mm_ss, encountered an hour overflow"));
      return 0;
    }
    assert(false);
  }
#endif // TEST_HAS_NO_EXCEPTIONS

  return 0;
}
