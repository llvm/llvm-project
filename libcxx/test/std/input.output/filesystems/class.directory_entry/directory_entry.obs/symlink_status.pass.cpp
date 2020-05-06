//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <filesystem>

// class directory_entry

// file_status symlink_status() const;
// file_status symlink_status(error_code&) const noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "filesystem_test_helper.h"
#include "rapid-cxx-test.h"

#include "test_macros.h"

TEST_SUITE(directory_entry_obs_suite)

TEST_CASE(test_signature) {
  using namespace fs;
  static_test_env static_env;
  {
    const directory_entry e("foo");
    std::error_code ec;
    static_assert(std::is_same<decltype(e.symlink_status()), file_status>::value, "");
    static_assert(std::is_same<decltype(e.symlink_status(ec)), file_status>::value, "");
    static_assert(noexcept(e.symlink_status()) == false, "");
    static_assert(noexcept(e.symlink_status(ec)) == true, "");
  }
  path TestCases[] = {static_env.File, static_env.Dir, static_env.SymlinkToFile,
                      static_env.DNE};
  for (const auto& p : TestCases) {
    const directory_entry e(p);
    std::error_code pec = GetTestEC(), eec = GetTestEC(1);
    file_status ps = fs::symlink_status(p, pec);
    file_status es = e.symlink_status(eec);
    TEST_CHECK(ps.type() == es.type());
    TEST_CHECK(ps.permissions() == es.permissions());
    TEST_CHECK(pec == eec);
  }
  for (const auto& p : TestCases) {
    const directory_entry e(p);
    file_status ps = fs::symlink_status(p);
    file_status es = e.symlink_status();
    TEST_CHECK(ps.type() == es.type());
    TEST_CHECK(ps.permissions() == es.permissions());
  }
}

TEST_SUITE_END()
