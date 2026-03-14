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

// Note there is no Standard way to change the remote database used.
// That is tested in
//   test/libcxx/time/time.zone/time.zone.db/time.zone.db.remote/reload_tzdb.pass.cpp

// const tzdb& reload_tzdb();

#include <cassert>
#include <chrono>
#include <iterator>

#include "test_macros.h"

int main(int, const char**) {
  const std::chrono::tzdb_list& list = std::chrono::get_tzdb_list();
  std::string version                = list.front().version;
  assert(!version.empty());

  assert(std::distance(list.begin(), list.end()) == 1);
  assert(std::distance(list.cbegin(), list.cend()) == 1);
  assert(std::chrono::remote_version() == version);

  std::chrono::reload_tzdb();

  assert(std::distance(list.begin(), list.end()) == 1);
  assert(std::distance(list.cbegin(), list.cend()) == 1);
  assert(std::chrono::remote_version() == version);

  return 0;
}
