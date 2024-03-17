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

// class time_zone_link;

// string_view name()   const noexcept;

#include <cassert>
#include <chrono>

#include "test_macros.h"

int main(int, const char**) {
  const std::chrono::tzdb& tzdb = std::chrono::get_tzdb();
  assert(tzdb.links.size() > 1);

  std::same_as<std::string_view> auto name = tzdb.links[0].name();
  static_assert(noexcept(tzdb.links[0].name()));
  assert(name != tzdb.links[1].name());

  return 0;
}
