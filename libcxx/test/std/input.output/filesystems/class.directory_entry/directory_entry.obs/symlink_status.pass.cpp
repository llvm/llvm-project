//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-create-symlinks
// UNSUPPORTED: c++03, c++11, c++14

// <filesystem>

// class directory_entry

// file_status symlink_status() const;
// file_status symlink_status(error_code&) const noexcept;

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "filesystem_test_helper.h"
#include "test_macros.h"
namespace fs = std::filesystem;

static void test_signature() {
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
    assert(ps.type() == es.type());
    assert(ps.permissions() == es.permissions());
    assert(pec == eec);
  }
  for (const auto& p : TestCases) {
    const directory_entry e(p);
    file_status ps = fs::symlink_status(p);
    file_status es = e.symlink_status();
    assert(ps.type() == es.type());
    assert(ps.permissions() == es.permissions());
  }
}

int main(int, char**) {
  test_signature();
  return 0;
}
