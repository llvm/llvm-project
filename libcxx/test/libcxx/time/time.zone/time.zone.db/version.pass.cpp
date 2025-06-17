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

// Tests the IANA database version parsing.
// This is not part of the public tzdb interface.

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string_view>
#include <string>

#include "assert_macros.h"
#include "concat_macros.h"
#include "filesystem_test_helper.h"
#include "test_tzdb.h"

scoped_test_env env;
[[maybe_unused]] const std::filesystem::path dir = env.create_dir("zoneinfo");
const std::filesystem::path data                 = env.create_file("zoneinfo/tzdata.zi");

std::string_view std::chrono::__libcpp_tzdb_directory() {
  static std::string result = dir.string();
  return result;
}

static void test(std::string_view input, std::string_view expected) {
  std::ofstream{data}.write(input.data(), input.size());
  std::string version = std::chrono::remote_version();

  TEST_REQUIRE(
      version == expected,
      TEST_WRITE_CONCATENATED(
          "\nInput            ", input, "\nExpected version ", expected, "\nActual version   ", version, '\n'));
}

static void test_exception(std::string_view input, [[maybe_unused]] std::string_view what) {
  std::ofstream{data}.write(input.data(), input.size());

  TEST_VALIDATE_EXCEPTION(
      std::runtime_error,
      [&]([[maybe_unused]] const std::runtime_error& e) {
        TEST_LIBCPP_REQUIRE(
            e.what() == what,
            TEST_WRITE_CONCATENATED("\nExpected exception ", what, "\nActual exception   ", e.what(), '\n'));
      },
      TEST_IGNORE_NODISCARD std::chrono::remote_version());
}

int main(int, const char**) {
  test_exception("", std::string{"corrupt tzdb: expected character '#', got '"} + (char)EOF + "' instead");
  test_exception("#version", "corrupt tzdb: expected whitespace");
  test("#version     \t                      ABCD", "ABCD");
  test("#Version     \t                      ABCD", "ABCD");
  test("#vErsion     \t                      ABCD", "ABCD");
  test("#verSion     \t                      ABCD", "ABCD");
  test("#VERSION     \t                      ABCD", "ABCD");
  test("#          \t   version      \t      2023a", "2023a");

  return 0;
}
