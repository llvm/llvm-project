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

// const string remote_version();

#include <chrono>

#include <cassert>

#include "test_macros.h"

int main(int, const char**) {
  std::string version = std::chrono::remote_version();
  assert(!version.empty());

  assert(version == std::chrono::get_tzdb().version);
  assert(version == std::chrono::get_tzdb_list().front().version);

  return 0;
}
