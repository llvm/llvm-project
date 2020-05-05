//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <filesystem>

// path weakly_canonical(const path& p);
// path weakly_canonical(const path& p, error_code& ec);

#include "filesystem_include.h"
#include <type_traits>
#include <vector>
#include <iostream>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "filesystem_test_helper.h"


int main(int, char**) {

  static_test_env static_env;

  // clang-format off
  struct {
    std::string input;
    std::string expect;
  } TestCases[] = {
      {"", fs::current_path()},
      {".", fs::current_path()},
      {"/", "/"},
      {"/foo", "/foo"},
      {"/.", "/"},
      {"/./", "/"},
      {"a/b", fs::current_path() / "a/b"},
      {"a", fs::current_path() / "a"},
      {"a/b/", fs::current_path() / "a/b/"},
      {static_env.File, static_env.File},
      {static_env.Dir, static_env.Dir},
      {static_env.SymlinkToDir, static_env.Dir},
      {static_env.SymlinkToDir / "dir2/.", static_env.Dir / "dir2"},
      // FIXME? If the trailing separator occurs in a part of the path that exists,
      // it is omitted. Otherwise it is added to the end of the result.
      {static_env.SymlinkToDir / "dir2/./", static_env.Dir / "dir2"},
      {static_env.SymlinkToDir / "dir2/DNE/./", static_env.Dir / "dir2/DNE/"},
      {static_env.SymlinkToDir / "dir2", static_env.Dir2},
      {static_env.SymlinkToDir / "dir2/../dir2/DNE/..", static_env.Dir2 / ""},
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
    fs::path p(TC.input);
    const fs::path output = fs::weakly_canonical(p);
    if (!PathEq(output, TC.expect)) {
      Failed = true;
      std::cerr << "TEST CASE #" << ID << " FAILED: \n";
      std::cerr << "  Input: '" << TC.input << "'\n";
      std::cerr << "  Expected: '" << TC.expect << "'\n";
      std::cerr << "  Output: '" << output.native() << "'";
      std::cerr << std::endl;
    }
  }
  return Failed;
}
