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
#include "src/__support/File/dir_scan_impl.h"
#include "src/__support/error_or.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

struct MockDirState {
  int read_errno_val = 0;
  int open_errno_val = 0;
  // how many successful reads until it fails.
  size_t read_fails_at = static_cast<size_t>(-1); // Don't fail by default
  size_t current_read_index = 0;
};

struct MockDir {
  inline static MockDirState state;

  inline static const char *test_files[] = {".", "..", "b.txt", "a.md",
                                            "c.pdf"};
  inline static size_t test_files_count =
      sizeof(MockDir::test_files) / sizeof(MockDir::test_files[0]);
  alignas(
      struct dirent) inline static char dirent_buffer[sizeof(struct dirent) +
                                                      256] = {};

  static LIBC_NAMESPACE::ErrorOr<MockDir *> open(const char *) {
    if (state.open_errno_val != 0)
      return LIBC_NAMESPACE::Error(state.open_errno_val);
    state.current_read_index = 0;
    return new MockDir();
  }

  LIBC_NAMESPACE::ErrorOr<struct dirent *> read() {
    if (state.current_read_index == state.read_fails_at) {
      return LIBC_NAMESPACE::Error(state.read_errno_val);
    }

    if (state.current_read_index >= test_files_count) {
      return nullptr;
    }

    struct dirent *entry = reinterpret_cast<struct dirent *>(dirent_buffer);
    entry->d_ino = state.current_read_index + 1;
    const char *name = test_files[state.current_read_index];
    size_t i = 0;
    while (name[i] != '\0') {
      entry->d_name[i] = name[i];
      ++i;
    }
    entry->d_name[i] = '\0';
    state.current_read_index++;
    return entry;
  }

  static size_t reclen(struct dirent *entry) {
    size_t name_len = 0;
    while (entry->d_name[name_len] != '\0')
      ++name_len;
    return sizeof(struct dirent) + name_len + 1;
  }

  int close() {
    delete this;
    return 0;
  }
};

} // namespace LIBC_NAMESPACE_DECL

class LlvmLibcScanImplTest : public LIBC_NAMESPACE::testing::Test {
protected:
  void SetUp() override {
    LIBC_NAMESPACE::MockDir::state = LIBC_NAMESPACE::MockDir::state =
        LIBC_NAMESPACE::MockDirState{};
  }
};

TEST_F(LlvmLibcScanImplTest, OpenFails) {

  struct dirent **namelist = nullptr;
  LIBC_NAMESPACE::MockDir::state.open_errno_val = EACCES;

  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, nullptr, nullptr);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), EACCES);
}

TEST_F(LlvmLibcScanImplTest, ReadFailsAtStart) {

  struct dirent **namelist = nullptr;
  LIBC_NAMESPACE::MockDir::state.read_errno_val = ENOENT;
  LIBC_NAMESPACE::MockDir::state.read_fails_at = 1;

  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, nullptr, nullptr);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), ENOENT);
}

TEST_F(LlvmLibcScanImplTest, ReadFailsMidway) {

  struct dirent **namelist = nullptr;
  LIBC_NAMESPACE::MockDir::state.read_errno_val = ENOENT;
  LIBC_NAMESPACE::MockDir::state.read_fails_at = 3;

  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, nullptr, nullptr);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), ENOENT);
}
