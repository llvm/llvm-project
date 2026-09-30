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
#include "src/__support/CPP/limits.h"
#include "src/__support/File/dir_scan_impl.h"
#include "src/__support/error_or.h"
#include "src/dirent/alphasort.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

struct MockDirTestSetup {
  int read_errno_val = 0;
  int open_errno_val = 0;
  // how many successful reads until it fails.
  size_t read_fails_at =
      cpp::numeric_limits<size_t>::max(); // Don't fail by default
  const char *test_files[5] = {".", "..", "b.txt", "a.md", "c.pdf"};
  size_t test_files_count = sizeof(test_files) / sizeof(test_files[0]);
};

MockDirTestSetup test_setup;

struct MockDir {
  alignas(struct dirent) char dirent_buffer[sizeof(struct dirent) + 256] = {};
  size_t current_read_index = 0;
  struct dirent *entry = reinterpret_cast<struct dirent *>(dirent_buffer);

  static LIBC_NAMESPACE::ErrorOr<MockDir *> open(const char *) {
    if (test_setup.open_errno_val != 0)
      return LIBC_NAMESPACE::Error(test_setup.open_errno_val);
    return new MockDir();
  }

  LIBC_NAMESPACE::ErrorOr<struct dirent *> read() {
    if (current_read_index == test_setup.read_fails_at)
      return LIBC_NAMESPACE::Error(test_setup.read_errno_val);

    if (current_read_index >= test_setup.test_files_count)
      return nullptr;

    entry->d_ino = current_read_index + 1;
    const char *name = test_setup.test_files[current_read_index];
    size_t i = 0;
    while (name[i] != '\0') {
      entry->d_name[i] = name[i];
      ++i;
    }
    entry->d_name[i] = '\0';
    current_read_index++;
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
    LIBC_NAMESPACE::testing::Test::SetUp();
    LIBC_NAMESPACE::test_setup = LIBC_NAMESPACE::MockDirTestSetup{};
  }
};

TEST_F(LlvmLibcScanImplTest, SuccessfulRun) {

  struct dirent **namelist = nullptr;
  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, nullptr, nullptr);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(),
            static_cast<int>(LIBC_NAMESPACE::test_setup.test_files_count));

  for (size_t i = 0; i < LIBC_NAMESPACE::test_setup.test_files_count; ++i) {
    ASSERT_STREQ(namelist[i]->d_name, LIBC_NAMESPACE::test_setup.test_files[i]);
  }
}

TEST_F(LlvmLibcScanImplTest, OpenFails) {

  struct dirent **namelist = nullptr;
  LIBC_NAMESPACE::test_setup.open_errno_val = EACCES;

  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, nullptr, nullptr);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), EACCES);
}

TEST_F(LlvmLibcScanImplTest, ReadFailsAtStart) {

  struct dirent **namelist = nullptr;
  LIBC_NAMESPACE::test_setup.read_errno_val = ENOENT;
  LIBC_NAMESPACE::test_setup.read_fails_at = 0;

  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, nullptr, nullptr);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), ENOENT);
}

TEST_F(LlvmLibcScanImplTest, ReadFailsMidway) {

  struct dirent **namelist = nullptr;
  LIBC_NAMESPACE::test_setup.read_errno_val = ENOENT;
  LIBC_NAMESPACE::test_setup.read_fails_at = 3;

  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, nullptr, nullptr);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), ENOENT);
}

int partialorder(const struct dirent **a, const struct dirent **b) {
  char char_a = (*a)->d_name[0];
  char char_b = (*b)->d_name[0];

  auto get_weight = [](char c) {
    if (c == 'a')
      return 1;
    if (c == 'b')
      return 2;
    return 3;
  };

  int weight_a = get_weight(char_a);
  int weight_b = get_weight(char_b);

  return weight_a - weight_b;
}

TEST_F(LlvmLibcScanImplTest, TestPartialOrdering) {

  struct dirent **namelist = nullptr;
  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, nullptr, partialorder);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(),
            static_cast<int>(LIBC_NAMESPACE::test_setup.test_files_count));
  int a_index = -1;
  int b_index = -1;
  for (int i = 0; i < res.value(); ++i) {
    if (namelist[i]->d_name[0] == 'a')
      a_index = i;

    else if (namelist[i]->d_name[0] == 'b') {
      b_index = i;
    }
  }
  ASSERT_NE(a_index, -1);
  ASSERT_NE(b_index, -1);
  ASSERT_GT(b_index, a_index);
}

int partialorder_reverse(const struct dirent **a, const struct dirent **b) {
  return -partialorder(a, b);
}

TEST_F(LlvmLibcScanImplTest, TestPartialOrderingReverse) {

  struct dirent **namelist = nullptr;
  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, nullptr, partialorder_reverse);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(),
            static_cast<int>(LIBC_NAMESPACE::test_setup.test_files_count));
  int a_index = -1;
  int b_index = -1;
  for (int i = 0; i < res.value(); ++i) {
    if (namelist[i]->d_name[0] == 'a')
      a_index = i;

    else if (namelist[i]->d_name[0] == 'b') {
      b_index = i;
    }
  }
  ASSERT_NE(a_index, -1);
  ASSERT_NE(b_index, -1);
  ASSERT_GT(a_index, b_index);
}

TEST_F(LlvmLibcScanImplTest, TestTotalOrderingAZ) {
  struct dirent **namelist = nullptr;
  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, nullptr, LIBC_NAMESPACE::alphasort);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(),
            static_cast<int>(LIBC_NAMESPACE::test_setup.test_files_count));
  const char *desired_order[] = {".", "..", "a.md", "b.txt", "c.pdf", nullptr};
  for (size_t i = 0; i < LIBC_NAMESPACE::test_setup.test_files_count &&
                     desired_order[i] != nullptr;
       ++i) {
    ASSERT_STREQ(namelist[i]->d_name, desired_order[i]);
  }
}

int omegasort(const struct dirent **a, const struct dirent **b) {
  return -LIBC_NAMESPACE::alphasort(a, b);
}

TEST_F(LlvmLibcScanImplTest, TestTotalOrderingZA) {
  struct dirent **namelist = nullptr;
  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, nullptr, omegasort);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(),
            static_cast<int>(LIBC_NAMESPACE::test_setup.test_files_count));
  const char *desired_order[] = {"c.pdf", "b.txt", "a.md", "..", ".", nullptr};
  for (size_t i = 0; i < LIBC_NAMESPACE::test_setup.test_files_count &&
                     desired_order[i] != nullptr;
       ++i) {
    ASSERT_STREQ(namelist[i]->d_name, desired_order[i]);
  }
}

int skip_hidden(const struct dirent *entry) { return entry->d_name[0] != '.'; }

TEST_F(LlvmLibcScanImplTest, TesetFilter) {
  struct dirent **namelist = nullptr;
  auto res = LIBC_NAMESPACE::internal::scan_impl<LIBC_NAMESPACE::MockDir>(
      "fake/path", &namelist, skip_hidden, nullptr);
  ASSERT_TRUE(res.has_value());
  size_t desired_count = LIBC_NAMESPACE::test_setup.test_files_count - 2;
  ASSERT_EQ(res.value(), static_cast<int>(desired_count));
  const char *desired_files[] = {"b.txt", "a.md", "c.pdf", nullptr};

  for (size_t i = 0; i < desired_count && desired_files[i] != nullptr; ++i)
    ASSERT_STREQ(namelist[i]->d_name, desired_files[i]);
}
