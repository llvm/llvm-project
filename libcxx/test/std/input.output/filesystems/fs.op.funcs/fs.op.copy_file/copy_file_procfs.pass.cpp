//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// REQUIRES: linux
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// bool copy_file(const path& from, const path& to);
// bool copy_file(const path& from, const path& to, error_code& ec) noexcept;
// bool copy_file(const path& from, const path& to, copy_options options);
// bool copy_file(const path& from, const path& to, copy_options options,
//           error_code& ec) noexcept;

#include <filesystem>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"

using namespace std::filesystem;

// Linux has various virtual filesystems such as /proc and /sys
// where files may have no length (st_size == 0), but still contain data.
// This is because the to-be-read data is usually generated ad-hoc by the reading syscall
// These files can not be copied with kernel-side copies like copy_file_range or sendfile,
// and must instead be copied via a traditional userspace read + write loop.
int main(int, char**) {
  const path procfile{"/proc/self/comm"};
  assert(file_size(procfile) == 0);

  scoped_test_env env;
  std::error_code ec = GetTestEC();

  const path dest = env.make_env_path("dest");

  assert(copy_file(procfile, dest, ec));
  assert(!ec);

  // a tad fragile if lit ever changes the output name,
  // but the easiest way to assert that the correct data was copied.
  assert(file_size(dest) == sizeof("t.tmp.exe"));
}
