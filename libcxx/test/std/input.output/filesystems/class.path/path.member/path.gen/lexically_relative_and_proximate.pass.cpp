//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// class path

// path lexically_relative(const path& p) const;
// path lexically_proximate(const path& p) const;

#include <filesystem>
#include <string>

#include "../../path_helper.h"
#include "assert_macros.h"
#include "concat_macros.h"
#include "count_new.h"
#include "test_macros.h"
namespace fs = std::filesystem;

int main(int, char**) {
  // clang-format off
  struct {
    std::string input;
    std::string base;
    std::string expect;
  } TestCases[] = {
      {"", "", "."},
      {"/", "a", ""},
      {"a", "/", ""},
      {"//net", "a", ""},
      {"a", "//net", ""},
#ifdef _WIN32
      {"//net/", "//net", ""},
      {"//net", "//net/", ""},
      {"C:\\a\\b", "C:/a", "b"},
#else
      {"//net/", "//net", "."},
      {"//net", "//net/", "."},
      {"C:\\a\\b", "C:/a", "../../C:\\a\\b"},
#endif
      {"//base", "a", ""},
      {"a", "a", "."},
      {"a/b", "a/b", "."},
      {"a/b/c/", "a/b/c/", "."},
      {"//net", "//net", "."},
      {"//net/", "//net/", "."},
      {"//net/a/b", "//net/a/b", "."},
      {"/a/d", "/a/b/c", "../../d"},
      {"/a/b/c", "/a/d", "../b/c"},
      {"a/b/c", "a", "b/c"},
      {"a/b/c", "a/b/c/x/y", "../.."},
      {"a/b/c", "a/b/c", "."},
      {"a/b", "c/d", "../../a/b"}
  };
  // clang-format on
  for (auto& TC : TestCases) {
    const fs::path p(TC.input);
    const fs::path output = p.lexically_relative(TC.base);
    fs::path expect(TC.expect);
    expect.make_preferred();

    // clang-format off
    TEST_REQUIRE(
        PathEq(output, expect),
        TEST_WRITE_CONCATENATED(
            "path::lexically_relative test case failed",
            "\nInput: ", TC.input,
            "\nBase: ", TC.base,
            "\nExpected: ", expect,
            "\nOutput: ", output));
    // clang-format on

    const fs::path proximate_output = p.lexically_proximate(TC.base);
    // [path.gen] lexically_proximate
    // Returns: If the value of lexically_relative(base) is not an empty path,
    // return it. Otherwise return *this.
    const fs::path proximate_expect = expect.empty() ? p : expect;

    // clang-format off
    TEST_REQUIRE(
        PathEq(proximate_output, proximate_expect),
        TEST_WRITE_CONCATENATED(
            "path::lexically_proximate test case failed",
            "\nInput: ", TC.input,
            "\nBase: ", TC.base,
            "\nExpected: ", proximate_expect,
            "\nOutput: ", proximate_output));
    // clang-format on
  }

  return 0;
}
