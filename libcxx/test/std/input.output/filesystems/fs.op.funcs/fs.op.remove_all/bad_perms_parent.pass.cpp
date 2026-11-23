//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-filesystem

// UNSUPPORTED: windows
// XFAIL: using-built-library-before-llvm-23

// Verify that remove_all reports the correct error (permission_denied)
// when the parent directory has insufficient permissions.

// <filesystem>

#include <filesystem>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;

int main(int, char**) {
  scoped_test_env env;

  const fs::path parent_dir = env.create_dir("parent");
  const fs::path child_dir  = env.create_dir(parent_dir / "child");
  permissions(parent_dir, fs::perms::owner_read | fs::perms::owner_write);

  const auto BadRet = static_cast<std::uintmax_t>(-1);
  std::error_code ec;
  assert(fs::remove_all(parent_dir, ec) == BadRet);
  assert(ec == std::errc::permission_denied);

  permissions(parent_dir, fs::perms::owner_all);
  return 0;
}
