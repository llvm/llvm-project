//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// path weakly_canonical(const path& p);
// path weakly_canonical(const path& p, error_code& ec);

#include "filesystem_include.h"
#include <cstdio>
#include <string>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "filesystem_test_helper.h"


int main(int, char**) {

  static_test_env static_env;

  fs::path root = fs::current_path().root_path();
  // clang-format off
  struct {
    fs::path input;
    fs::path expect;
  } TestCases[] = {
      {"", fs::current_path()},
      {".", fs::current_path()},
      {"/", root},
      {"/foo", root / "foo"},
      {"/.", root},
      {"/./", root},
      {"a/b", fs::current_path() / "a/b"},
      {"a", fs::current_path() / "a"},
      {"a/b/", fs::current_path() / "a/b/"},
      {static_env.File, static_env.File},
      {static_env.Dir, static_env.Dir},
      {static_env.SymlinkToDir, static_env.Dir},
      {static_env.SymlinkToDir / "dir2/.", static_env.Dir / "dir2"},
      // Note: If the trailing separator occurs in a part of the path that exists,
      // it is omitted. Otherwise it is added to the end of the result.
      // MS STL and libstdc++ behave similarly.
      {static_env.SymlinkToDir / "dir2/./", static_env.Dir / "dir2"},
      {static_env.SymlinkToDir / "dir2/DNE/./", static_env.Dir / "dir2/DNE/"},
      {static_env.SymlinkToDir / "dir2", static_env.Dir2},
#ifdef _WIN32
      // On Windows, this path is considered to exist (even though it
      // passes through a nonexistent directory), and thus is returned
      // without a trailing slash, see the note above.
      {static_env.SymlinkToDir / "dir2/../dir2/DNE/..", static_env.Dir2},
#else
      {static_env.SymlinkToDir / "dir2/../dir2/DNE/..", static_env.Dir2 / ""},
#endif
      {static_env.SymlinkToDir / "dir2/dir3/../DNE/DNE2", static_env.Dir2 / "DNE/DNE2"},
      {static_env.Dir / "../dir1", static_env.Dir},
      {static_env.Dir / "./.", static_env.Dir},
      {static_env.Dir / "DNE/../foo", static_env.Dir / "foo"}
  };
  // clang-format on
  int ID = 0;
  bool Failed = false;
  for (auto& TC : TestCases) {
    ++ID;
    fs::path p = TC.input;
    fs::path expect = TC.expect;
    expect.make_preferred();
    const fs::path output = fs::weakly_canonical(p);
    if (!PathEq(output, expect)) {
      Failed = true;
      std::fprintf(stderr, "TEST CASE #%d FAILED:\n"
                  "  Input: '%s'\n"
                  "  Expected: '%s'\n"
                  "  Output: '%s'\n",
        ID, TC.input.string().c_str(), expect.string().c_str(),
        output.string().c_str());
    }
  }
  return Failed;
}
