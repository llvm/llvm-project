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
//
// class tzdb_list;
//
// const_iterator erase_after(const_iterator p);

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
  write("# version 1");
  std::chrono::tzdb_list& list = std::chrono::get_tzdb_list(); // [1]

  write("# version 2");
  std::chrono::reload_tzdb(); // [2, 1]

  assert(std::distance(list.begin(), list.end()) == 2);
  assert(list.front().version == "2");

  list.erase_after(list.begin()); // [2]
  assert(std::distance(list.begin(), list.end()) == 1);
  assert(list.front().version == "2");

  write("# version 3");
  std::chrono::reload_tzdb(); // [3, 2]
  assert(std::distance(list.begin(), list.end()) == 2);

  write("# version 4");
  std::chrono::reload_tzdb(); // [4, 3, 2]
  assert(std::distance(list.begin(), list.end()) == 3);
  assert(list.front().version == "4");

  std::chrono::tzdb_list::const_iterator it = ++list.begin();
  assert(it->version == "3");

  list.erase_after(it); // [4, 3]
  assert(std::distance(list.begin(), list.end()) == 2);
  assert(list.front().version == "4");
  assert(it->version == "3");
}
