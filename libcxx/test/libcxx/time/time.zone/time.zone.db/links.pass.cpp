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

// Tests the IANA database rules links and operations.
// This is not part of the public tzdb interface.

#include <chrono>

#include <cassert>
#include <chrono>
#include <fstream>
#include <string>
#include <string_view>

#include "assert_macros.h"
#include "concat_macros.h"
#include "filesystem_test_helper.h"
#include "test_tzdb.h"

scoped_test_env env;
[[maybe_unused]] const std::filesystem::path dir = env.create_dir("zoneinfo");
const std::filesystem::path file                 = env.create_file("zoneinfo/tzdata.zi");

std::string_view std::chrono::__libcpp_tzdb_directory() {
  static std::string result = dir.string();
  return result;
}

void write(std::string_view input) {
  static int version = 0;

  std::ofstream f{file};
  f << "# version " << version++ << '\n';
  f.write(input.data(), input.size());
}

static const std::chrono::tzdb& parse(std::string_view input) {
  write(input);
  return std::chrono::reload_tzdb();
}

static void test_exception(std::string_view input, [[maybe_unused]] std::string_view what) {
  write(input);

  TEST_VALIDATE_EXCEPTION(
      std::runtime_error,
      [&]([[maybe_unused]] const std::runtime_error& e) {
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
      },
      TEST_IGNORE_NODISCARD std::chrono::reload_tzdb());
}

static void test_invalid() {
  test_exception("L", "corrupt tzdb: expected whitespace");

  test_exception("L ", "corrupt tzdb: expected a string");

  test_exception("L n", "corrupt tzdb: expected whitespace");

  test_exception("L n ", "corrupt tzdb: expected a string");
}

static void test_link() {
  const std::chrono::tzdb& result = parse(
      R"(
L z d
l b a
lInK b b
)");
  assert(result.links.size() == 3);

  assert(result.links[0].name() == "a");
  assert(result.links[0].target() == "b");

  assert(result.links[1].name() == "b");
  assert(result.links[1].target() == "b");

  assert(result.links[2].name() == "d");
  assert(result.links[2].target() == "z");
}

int main(int, const char**) {
  test_invalid();
  test_link();

  return 0;
}
