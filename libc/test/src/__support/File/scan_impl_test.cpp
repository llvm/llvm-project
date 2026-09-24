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
#include "src/__support/File/scan_impl.h"
#include "src/__support/error_or.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

struct MockDir {
  static int read_errno_val;
  static int read_fails_at; // how many successful reads until it fails.

  static struct dirent dummy_entry;

  static LIBC_NAMESPACE::ErrorOr<MockDir *> open(const char *path) {
    (void)path;
    return new MockDir();
  }

  LIBC_NAMESPACE::ErrorOr<struct dirent *> read() {
    read_fails_at--;
    if (read_fails_at <= 0) {
      return LIBC_NAMESPACE::Error(read_errno_val);
    }

    dummy_entry.d_name[0] = 'a';
    dummy_entry.d_name[1] = '\0';
    return &dummy_entry;
  }

  static size_t reclen(struct dirent *) { return sizeof(struct dirent); }

  int close() {
    delete this;
    return 0;
  }
};

int MockDir::read_errno_val = 0;
int MockDir::read_fails_at = 1;
struct dirent MockDir::dummy_entry = {};

} // namespace LIBC_NAMESPACE_DECL

TEST(LlvmLibcScanImplTest, ReadFailsAtStart) {

  struct dirent **namelist = nullptr;
  LIBC_NAMESPACE::MockDir::read_errno_val = ENOENT;
  LIBC_NAMESPACE::MockDir::read_fails_at = 1;

  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, nullptr, nullptr);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), ENOENT);
}

TEST(LlvmLibcScanImplTest, ReadFailsMidway) {

  struct dirent **namelist = nullptr;
  LIBC_NAMESPACE::MockDir::read_errno_val = ENOENT;
  LIBC_NAMESPACE::MockDir::read_fails_at = 3;

  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, nullptr, nullptr);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), ENOENT);
}
