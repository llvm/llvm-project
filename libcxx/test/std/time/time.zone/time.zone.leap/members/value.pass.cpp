//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-incomplete-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

// class leap_second;

// constexpr seconds value() const noexcept;

#include <cassert>
#include <chrono>

#include "test_macros.h"

// Add the include path required by test_chrono_leap_second.h when using libc++.
// ADDITIONAL_COMPILE_FLAGS(stdlib=libc++): -I %S/../../../../../../src/include
#include "test_chrono_leap_second.h"

constexpr void test(const std::chrono::leap_second leap_second, std::chrono::seconds expected) {
  std::same_as<std::chrono::seconds> auto value = leap_second.value();
  assert(value == expected);
  static_assert(noexcept(leap_second.value()));
}

constexpr bool test() {
  test(test_leap_second_create(std::chrono::sys_seconds{std::chrono::seconds{0}}, std::chrono::seconds{1}),
       std::chrono::seconds{1});

  return true;
}

int main(int, const char**) {
  test();
  static_assert(test());

  // test with the real tzdb
  const std::chrono::tzdb& tzdb = std::chrono::get_tzdb();
  assert(!tzdb.leap_seconds.empty());
  test(tzdb.leap_seconds[0], tzdb.leap_seconds[0].value());

  return 0;
}
