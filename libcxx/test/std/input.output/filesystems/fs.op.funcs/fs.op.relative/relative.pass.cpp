//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FILE_DEPENDENCIES: ../../Inputs/static_test_env
// UNSUPPORTED: c++98, c++03

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
  fs::path p(StaticEnv::File);
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(StaticEnv::File));
}

TEST_CASE(test_signature_3) {
  fs::path p(StaticEnv::Dir);
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(StaticEnv::Dir));
}

TEST_CASE(test_signature_4) {
  fs::path p(StaticEnv::SymlinkToDir);
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(StaticEnv::Dir));
}

TEST_CASE(test_signature_5) {
  fs::path p(StaticEnv::SymlinkToDir / "dir2/.");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(StaticEnv::Dir / "dir2"));
}

TEST_CASE(test_signature_6) {
  // FIXME? If the trailing separator occurs in a part of the path that exists,
  // it is omitted. Otherwise it is added to the end of the result.
  fs::path p(StaticEnv::SymlinkToDir / "dir2/./");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(StaticEnv::Dir / "dir2"));
}

TEST_CASE(test_signature_7) {
  fs::path p(StaticEnv::SymlinkToDir / "dir2/DNE/./");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(StaticEnv::Dir / "dir2/DNE/"));
}

TEST_CASE(test_signature_8) {
  fs::path p(StaticEnv::SymlinkToDir / "dir2");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(StaticEnv::Dir2));
}

TEST_CASE(test_signature_9) {
  fs::path p(StaticEnv::SymlinkToDir / "dir2/../dir2/DNE/..");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(StaticEnv::Dir2 / ""));
}

TEST_CASE(test_signature_10) {
  fs::path p(StaticEnv::SymlinkToDir / "dir2/dir3/../DNE/DNE2");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(StaticEnv::Dir2 / "DNE/DNE2"));
}

TEST_CASE(test_signature_11) {
  fs::path p(StaticEnv::Dir / "../dir1");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(StaticEnv::Dir));
}

TEST_CASE(test_signature_12) {
  fs::path p(StaticEnv::Dir / "./.");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(StaticEnv::Dir));
}

TEST_CASE(test_signature_13) {
  fs::path p(StaticEnv::Dir / "DNE/../foo");
  const fs::path output = fs::weakly_canonical(p);
  TEST_CHECK(output == std::string(StaticEnv::Dir / "foo"));
}

TEST_SUITE_END()
