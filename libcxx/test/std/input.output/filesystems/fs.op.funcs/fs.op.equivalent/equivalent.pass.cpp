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

// Starting in Android N (API 24), SELinux policy prevents the shell user from
// creating a hard link.
// XFAIL: LIBCXX-ANDROID-FIXME && !android-device-api={{21|22|23}}

// <filesystem>

// bool equivalent(path const& lhs, path const& rhs);
// bool equivalent(path const& lhs, path const& rhs, std::error_code& ec) noexcept;

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "assert_macros.h"
#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void signature_test() {
  const path p;
  ((void)p);
  std::error_code ec;
  ((void)ec);
  ASSERT_NOEXCEPT(equivalent(p, p, ec));
  ASSERT_NOT_NOEXCEPT(equivalent(p, p));
}

static void equivalent_test() {
  static_test_env static_env;
  struct TestCase {
    path lhs;
    path rhs;
    bool expect;
  };
  const TestCase testCases[] = {
      {static_env.Dir, static_env.Dir, true},
      {static_env.File, static_env.Dir, false},
      {static_env.Dir, static_env.SymlinkToDir, true},
      {static_env.Dir, static_env.SymlinkToFile, false},
      {static_env.File, static_env.File, true},
      {static_env.File, static_env.SymlinkToFile, true},
  };
  for (auto& TC : testCases) {
    std::error_code ec;
    assert(equivalent(TC.lhs, TC.rhs, ec) == TC.expect);
    assert(!ec);
  }
}

static void equivalent_reports_error_if_input_dne() {
  static_test_env static_env;
  const path E = static_env.File;
  const path DNE = static_env.DNE;
  { // Test that an error is reported when either of the paths don't exist
    std::error_code ec = GetTestEC();
    assert(equivalent(E, DNE, ec) == false);
    assert(ec);
    assert(ec != GetTestEC());
  }
  {
    std::error_code ec = GetTestEC();
    assert(equivalent(DNE, E, ec) == false);
    assert(ec);
    assert(ec != GetTestEC());
  }
  {
    TEST_THROWS_TYPE(filesystem_error, equivalent(DNE, E));
    TEST_THROWS_TYPE(filesystem_error, equivalent(E, DNE));
  }
  { // Test that an exception is thrown if both paths do not exist.
    TEST_THROWS_TYPE(filesystem_error, equivalent(DNE, DNE));
  }
  {
    std::error_code ec = GetTestEC();
    assert(equivalent(DNE, DNE, ec) == false);
    assert(ec);
    assert(ec != GetTestEC());
  }
}

static void equivalent_hardlink_succeeds() {
  scoped_test_env env;
  path const file = env.create_file("file", 42);
  const path hl1 = env.create_hardlink(file, "hl1");
  const path hl2 = env.create_hardlink(file, "hl2");
  assert(equivalent(file, hl1));
  assert(equivalent(file, hl2));
  assert(equivalent(hl1, hl2));
}

#ifndef _WIN32
static void equivalent_is_other_succeeds() {
  scoped_test_env env;
  path const file = env.create_file("file", 42);
  const path fifo1 = env.create_fifo("fifo1");
  const path fifo2 = env.create_fifo("fifo2");
  // Required to test behavior for inputs where is_other(p) is true.
  assert(is_other(fifo1));
  assert(!equivalent(file, fifo1));
  assert(!equivalent(fifo2, file));
  assert(!equivalent(fifo1, fifo2));
  assert(equivalent(fifo1, fifo1));
}
#endif // _WIN32

int main(int, char**) {
  signature_test();
  equivalent_test();
  equivalent_reports_error_if_input_dne();
  equivalent_hardlink_succeeds();
#ifndef _WIN32
  equivalent_is_other_succeeds();
#endif

  return 0;
}
