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

// file_status status() const;
// file_status status(error_code const&) const noexcept;

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "filesystem_test_helper.h"
#include "test_macros.h"
namespace fs = std::filesystem;

static void test_basic() {
  using namespace fs;
  static_test_env static_env;
  {
    const fs::directory_entry e("foo");
    std::error_code ec;
    static_assert(std::is_same<decltype(e.status()), fs::file_status>::value, "");
    static_assert(std::is_same<decltype(e.status(ec)), fs::file_status>::value, "");
    static_assert(noexcept(e.status()) == false, "");
    static_assert(noexcept(e.status(ec)) == true, "");
  }
  path TestCases[] = {static_env.File, static_env.Dir, static_env.SymlinkToFile,
                      static_env.DNE};
  for (const auto& p : TestCases) {
    const directory_entry e(p);
    std::error_code pec = GetTestEC(), eec = GetTestEC(1);
    file_status ps = fs::status(p, pec);
    file_status es = e.status(eec);
    assert(ps.type() == es.type());
    assert(ps.permissions() == es.permissions());
    assert(pec == eec);
  }
  for (const auto& p : TestCases) {
    const directory_entry e(p);
    file_status ps = fs::status(p);
    file_status es = e.status();
    assert(ps.type() == es.type());
    assert(ps.permissions() == es.permissions());
  }
}

int main(int, char**) {
  test_basic();

  return 0;
}
