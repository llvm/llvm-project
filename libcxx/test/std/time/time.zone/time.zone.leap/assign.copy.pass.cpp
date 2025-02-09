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
//   leap_second& operator=(const leap_second&) = default;
//
//   ...
// };

#include <chrono>
#include <concepts>
#include <memory>
#include <type_traits>
#include <cassert>

#include "test_chrono_leap_second.h"

constexpr bool test() {
  std::chrono::leap_second a =
      test_leap_second_create(std::chrono::sys_seconds{std::chrono::seconds{0}}, std::chrono::seconds{1});
  std::chrono::leap_second b =
      test_leap_second_create(std::chrono::sys_seconds{std::chrono::seconds{10}}, std::chrono::seconds{15});

  //  operator== only compares the date member.
  assert(a.date() != b.date());
  assert(a.value() != b.value());

  {
    std::same_as<std::chrono::leap_second&> decltype(auto) result(b = a);
    assert(std::addressof(result) == std::addressof(b));

    assert(a.date() == b.date());
    assert(a.value() == b.value());
  }

  {
    // Tests an rvalue uses the copy assignment.
    std::same_as<std::chrono::leap_second&> decltype(auto) result(b = std::move(a));
    assert(std::addressof(result) == std::addressof(b));

    assert(a.date() == b.date());
    assert(a.value() == b.value());
  }

  return true;
}

int main(int, const char**) {
  static_assert(std::is_copy_assignable_v<std::chrono::leap_second>);

  test();
  static_assert(test());

  return 0;
}
