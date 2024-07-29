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

// enum class choose;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  using E = std::chrono::choose;
  static_assert(std::is_enum_v<E>);

  // Check that E is a scoped enum by checking for conversions.
  using UT = std::underlying_type_t<E>;
  static_assert(!std::is_convertible_v<E, UT>);

  [[maybe_unused]] const E& early = E::earliest;
  [[maybe_unused]] const E& late  = E::latest;

  return 0;
}
