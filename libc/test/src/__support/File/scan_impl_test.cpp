//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests templated implementation of the scandir logic.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_dirent.h"
#include "src/__support/error_or.h"
#include "src/__support/File/scan_impl.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

struct MockDir {
  static int read_errno_val;

  static LIBC_NAMESPACE::ErrorOr<MockDir *> open(const char *path) {
    (void)path;
    return new MockDir();
  }

  LIBC_NAMESPACE::ErrorOr<struct dirent *> read() {

    return LIBC_NAMESPACE::Error(read_errno_val);
  }

  int close() {
    delete this;
    return 0;
  }
};

int MockDir::read_errno_val = 0;

} // namespace LIBC_NAMESPACE_DECL

TEST(LlvmLibcScanImplTest, ReadFailsWithENOENT) {
  struct dirent **namelist = nullptr;
  LIBC_NAMESPACE::MockDir::read_errno_val = ENOENT;
  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, nullptr, nullptr);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), ENOENT);
}
