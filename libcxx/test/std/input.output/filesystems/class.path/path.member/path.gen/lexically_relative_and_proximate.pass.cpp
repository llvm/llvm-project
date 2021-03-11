//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class path

// path lexically_relative(const path& p) const;
// path lexically_proximate(const path& p) const;

#include "filesystem_include.h"
#include <cstdio>
#include <string>

#include "test_macros.h"
#include "count_new.h"
#include "filesystem_test_helper.h"


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
#else
      {"//net/", "//net", "."},
      {"//net", "//net/", "."},
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
  int ID = 0;
  bool Failed = false;
  for (auto& TC : TestCases) {
    ++ID;
    const fs::path p(TC.input);
    const fs::path output = p.lexically_relative(TC.base);
    fs::path expect(TC.expect);
    expect.make_preferred();
    auto ReportErr = [&](const char* Testing, fs::path const& Output,
                                              fs::path const& Expected) {
      Failed = true;
      std::fprintf(stderr, "TEST CASE #%d FAILED:\n"
                  "  Testing: %s\n"
                  "  Input: '%s'\n"
                  "  Base: '%s'\n"
                  "  Expected: '%s'\n"
                  "  Output: '%s'\n",
        ID, Testing, TC.input.c_str(), TC.base.c_str(),
        Expected.string().c_str(), Output.string().c_str());
    };
    if (!PathEq(output, expect))
      ReportErr("path::lexically_relative", output, expect);
    const fs::path proximate_output = p.lexically_proximate(TC.base);
    // [path.gen] lexically_proximate
    // Returns: If the value of lexically_relative(base) is not an empty path,
    // return it. Otherwise return *this.
    const fs::path proximate_expect = expect.empty() ? p : expect;
    if (!PathEq(proximate_output, proximate_expect))
      ReportErr("path::lexically_proximate", proximate_output, proximate_expect);
  }
  return Failed;
}
