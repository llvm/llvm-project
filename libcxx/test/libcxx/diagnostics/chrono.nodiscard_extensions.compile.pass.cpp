//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that format functions aren't marked [[nodiscard]] when
// _LIBCPP_DISABLE_NODISCARD_EXT is defined

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-incomplete-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_NODISCARD_EXT

#include <chrono>

#include "test_macros.h"

void test() {
  std::chrono::tzdb_list& list = std::chrono::get_tzdb_list();
  list.front();
  list.begin();
  list.end();
  list.cbegin();
  list.cend();

  std::chrono::get_tzdb_list();
  std::chrono::get_tzdb();
  std::chrono::remote_version();
}
