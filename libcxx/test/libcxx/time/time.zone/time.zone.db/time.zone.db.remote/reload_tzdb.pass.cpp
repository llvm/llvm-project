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

// const tzdb& reload_tzdb();

#include <cassert>
#include <chrono>
#include <fstream>
#include <iterator>

#include "filesystem_test_helper.h"
#include "test_macros.h"
#include "test_tzdb.h"

scoped_test_env env;
[[maybe_unused]] const std::filesystem::path dir = env.create_dir("zoneinfo");
const std::filesystem::path data                 = env.create_file("zoneinfo/tzdata.zi");

std::string_view std::chrono::__libcpp_tzdb_directory() {
  static std::string result = dir.string();
  return result;
}

static void write(std::string_view input) { std::ofstream{data}.write(input.data(), input.size()); }

int main(int, const char**) {
  write("# version old_version");
  const std::chrono::tzdb_list& list = std::chrono::get_tzdb_list();
  std::string version                = "new_version";

  assert(list.front().version == "old_version");
  assert(std::distance(list.begin(), list.end()) == 1);
  assert(std::distance(list.cbegin(), list.cend()) == 1);

  write("# version new_version");
  assert(std::chrono::remote_version() == version);

  std::chrono::reload_tzdb();

  assert(std::distance(list.begin(), list.end()) == 2);
  assert(std::distance(list.cbegin(), list.cend()) == 2);
  assert(list.front().version == version);

  return 0;
}
