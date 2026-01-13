//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-create-symlinks
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-filesystem

// <filesystem>

// path proximate(const path& p, error_code &ec)
// path proximate(const path& p, const path& base = current_path())
// path proximate(const path& p, const path& base, error_code& ec);

#include <filesystem>
#include <string>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;

static void test_signature_0() {
  fs::path p("");
  const fs::path output = fs::weakly_canonical(p);
  assert(output == fs::path::string_type(fs::current_path()));
}

static void test_signature_1() {
  fs::path p(".");
  const fs::path output = fs::weakly_canonical(p);
  assert(output == fs::path::string_type(fs::current_path()));
}

static void test_signature_2() {
  static_test_env static_env;
  fs::path p(static_env.File);
  const fs::path output = fs::weakly_canonical(p);
  assert(output == fs::path::string_type(static_env.File));
}

static void test_signature_3() {
  static_test_env static_env;
  fs::path p(static_env.Dir);
  const fs::path output = fs::weakly_canonical(p);
  assert(output == fs::path::string_type(static_env.Dir));
}

static void test_signature_4() {
  static_test_env static_env;
  fs::path p(static_env.SymlinkToDir);
  const fs::path output = fs::weakly_canonical(p);
  assert(output == fs::path::string_type(static_env.Dir));
}

static void test_signature_5() {
  static_test_env static_env;
  fs::path p(static_env.SymlinkToDir / "dir2/.");
  const fs::path output = fs::weakly_canonical(p);
  assert(output == fs::path::string_type(static_env.Dir / "dir2"));
}

static void test_signature_6() {
  static_test_env static_env;
  // FIXME? If the trailing separator occurs in a part of the path that exists,
  // it is omitted. Otherwise it is added to the end of the result.
  fs::path p(static_env.SymlinkToDir / "dir2/./");
  const fs::path output = fs::weakly_canonical(p);
  assert(output == fs::path::string_type(static_env.Dir / "dir2"));
}

static void test_signature_7() {
  static_test_env static_env;
  fs::path p(static_env.SymlinkToDir / "dir2/DNE/./");
  const fs::path output = fs::weakly_canonical(p);
  assert(output == fs::path::string_type(static_env.Dir / "dir2/DNE/"));
}

static void test_signature_8() {
  static_test_env static_env;
  fs::path p(static_env.SymlinkToDir / "dir2");
  const fs::path output = fs::weakly_canonical(p);
  assert(output == fs::path::string_type(static_env.Dir2));
}

static void test_signature_9() {
  static_test_env static_env;
  fs::path p(static_env.SymlinkToDir / "dir2/../dir2/DNE/..");
  const fs::path output = fs::weakly_canonical(p);
  // weakly_canonical has a quirk - if the path is considered to exist,
  // it's returned without a trailing slash, otherwise it's returned with
  // one (see a note in fs.op.weakly_canonical/weakly_canonical.pass.cpp).
  // On Windows, a path like existent/nonexistentsubdir/.. is considered
  // to exist, on posix it's considered to not exist. Therefore, the
  // result here differs in the trailing slash.
#ifdef _WIN32
  assert(output == fs::path::string_type(static_env.Dir2));
#else
  assert(output == fs::path::string_type(static_env.Dir2 / ""));
#endif
}

static void test_signature_10() {
  static_test_env static_env;
  fs::path p(static_env.SymlinkToDir / "dir2/dir3/../DNE/DNE2");
  const fs::path output = fs::weakly_canonical(p);
  assert(output == fs::path::string_type(static_env.Dir2 / "DNE/DNE2"));
}

static void test_signature_11() {
  static_test_env static_env;
  fs::path p(static_env.Dir / "../dir1");
  const fs::path output = fs::weakly_canonical(p);
  assert(output == fs::path::string_type(static_env.Dir));
}

static void test_signature_12() {
  static_test_env static_env;
  fs::path p(static_env.Dir / "./.");
  const fs::path output = fs::weakly_canonical(p);
  assert(output == fs::path::string_type(static_env.Dir));
}

static void test_signature_13() {
  static_test_env static_env;
  fs::path p(static_env.Dir / "DNE/../foo");
  const fs::path output = fs::weakly_canonical(p);
  assert(output == fs::path::string_type(static_env.Dir / "foo"));
}

int main(int, char**) {
  test_signature_0();
  test_signature_1();
  test_signature_2();
  test_signature_3();
  test_signature_4();
  test_signature_5();
  test_signature_6();
  test_signature_7();
  test_signature_8();
  test_signature_9();
  test_signature_10();
  test_signature_11();
  test_signature_12();
  test_signature_13();

  return 0;
}
