//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// path proximate(const path& p, error_code &ec)
// path proximate(const path& p, const path& base = current_path())
// path proximate(const path& p, const path& base, error_code& ec);

#include "filesystem_include.h"
#include <string>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"


TEST_SUITE(filesystem_proximate_path_test_suite)

TEST_CASE(test_signature_0) {
  fs::path p("");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(fs::current_path()));
}

TEST_CASE(test_signature_1) {
  fs::path p(".");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(fs::current_path()));
}

TEST_CASE(test_signature_2) {
  static_test_env static_env;
  fs::path p(static_env.File);
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(static_env.File));
}

TEST_CASE(test_signature_3) {
  static_test_env static_env;
  fs::path p(static_env.Dir);
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(static_env.Dir));
}

TEST_CASE(test_signature_4) {
  static_test_env static_env;
  fs::path p(static_env.SymlinkToDir);
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(static_env.Dir));
}

TEST_CASE(test_signature_5) {
  static_test_env static_env;
  fs::path p(static_env.SymlinkToDir / "dir2/.");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(static_env.Dir / "dir2"));
}

TEST_CASE(test_signature_6) {
  static_test_env static_env;
  // FIXME? If the trailing separator occurs in a part of the path that exists,
  // it is omitted. Otherwise it is added to the end of the result.
  fs::path p(static_env.SymlinkToDir / "dir2/./");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(static_env.Dir / "dir2"));
}

TEST_CASE(test_signature_7) {
  static_test_env static_env;
  fs::path p(static_env.SymlinkToDir / "dir2/DNE/./");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(static_env.Dir / "dir2/DNE/"));
}

TEST_CASE(test_signature_8) {
  static_test_env static_env;
  fs::path p(static_env.SymlinkToDir / "dir2");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(static_env.Dir2));
}

TEST_CASE(test_signature_9) {
  static_test_env static_env;
  fs::path p(static_env.SymlinkToDir / "dir2/../dir2/DNE/..");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(static_env.Dir2 / ""));
}

TEST_CASE(test_signature_10) {
  static_test_env static_env;
  fs::path p(static_env.SymlinkToDir / "dir2/dir3/../DNE/DNE2");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(static_env.Dir2 / "DNE/DNE2"));
}

TEST_CASE(test_signature_11) {
  static_test_env static_env;
  fs::path p(static_env.Dir / "../dir1");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(static_env.Dir));
}

TEST_CASE(test_signature_12) {
  static_test_env static_env;
  fs::path p(static_env.Dir / "./.");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(static_env.Dir));
}

TEST_CASE(test_signature_13) {
  static_test_env static_env;
  fs::path p(static_env.Dir / "DNE/../foo");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(static_env.Dir / "foo"));
}

TEST_SUITE_END()
