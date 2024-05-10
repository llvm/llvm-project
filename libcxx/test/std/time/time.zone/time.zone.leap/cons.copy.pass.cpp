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

// class leap_second
// {
//   leap_second(const leap_second&)            = default;
//
//   ...
// };

#include <chrono>
#include <concepts>
#include <cassert>

#include "test_chrono_leap_second.h"

constexpr bool test() {
  std::chrono::leap_second a =
      test_leap_second_create(std::chrono::sys_seconds{std::chrono::seconds{0}}, std::chrono::seconds{1});

  {
    std::chrono::leap_second b = a;

    //  operator== only compares the date member.
    assert(a.date() == b.date());
    assert(a.value() == b.value());
  }

#ifdef _LIBCPP_VERSION
  {
    // Tests an rvalue uses the copy constructor.
    // Since implementations are allowed to add additional constructors this is
    // a libc++ specific test.
    std::chrono::leap_second b = std::move(a);

    //  operator== only compares the date member.
    assert(a.date() == b.date());
    assert(a.value() == b.value());
  }
  // libc++ does not provide a default constructor.
  static_assert(!std::is_default_constructible_v<std::chrono::leap_second>);
#endif // _LIBCPP_VERSION

  return true;
}

int main(int, const char**) {
  static_assert(std::copy_constructible<std::chrono::leap_second>);

  test();
  static_assert(test());

  return 0;
}
