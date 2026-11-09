//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for the templated implementation of the scandir logic.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/struct_dirent.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/dir_scan_impl.h"
#include "src/__support/error_or.h"
#include "src/dirent/alphasort.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "test/UnitTest/Test.h"

namespace {

constexpr const char *TEST_FILES[] = {".", "..", "b.txt", "a.md", "c.pdf"};
constexpr size_t TEST_FILES_COUNT = sizeof(TEST_FILES) / sizeof(TEST_FILES[0]);

struct MockDirTestConfig {
  int read_errno_val = 0;
  int open_errno_val = 0;
  size_t read_fails_at = LIBC_NAMESPACE::cpp::numeric_limits<size_t>::max();
  const char *const *files = TEST_FILES;
  size_t files_count = TEST_FILES_COUNT;
  bool was_closed = false;
};

MockDirTestConfig g_mock_config;

struct MockDir {
  alignas(struct dirent) char dirent_buffer[sizeof(struct dirent) + 256] = {};
  size_t current_read_index = 0;

  static MockDir instance;

  static LIBC_NAMESPACE::ErrorOr<MockDir *> open(const char *) {
    if (g_mock_config.open_errno_val != 0)
      return LIBC_NAMESPACE::Error(g_mock_config.open_errno_val);
    instance.current_read_index = 0;
    instance.dirent_buffer[0] = '\0';
    g_mock_config.was_closed = false;
    return &instance;
  }

  LIBC_NAMESPACE::ErrorOr<struct dirent *> read() {
    if (current_read_index == g_mock_config.read_fails_at)
      return LIBC_NAMESPACE::Error(g_mock_config.read_errno_val);

    if (current_read_index >= g_mock_config.files_count)
      return nullptr;

    auto *entry = reinterpret_cast<struct dirent *>(dirent_buffer);
    entry->d_ino = current_read_index + 1;
    const char *name = g_mock_config.files[current_read_index];
    size_t name_len = LIBC_NAMESPACE::cpp::string_view(name).size();
    LIBC_NAMESPACE::inline_memcpy(entry->d_name, name, name_len + 1);
    current_read_index++;
    return entry;
  }

  static constexpr size_t reclen(const struct dirent *entry) {
    size_t name_len = 0;
    while (entry->d_name[name_len] != '\0')
      ++name_len;
    return sizeof(struct dirent) + name_len + 1;
  }

  int close() {
    g_mock_config.was_closed = true;
    return 0;
  }
};

MockDir MockDir::instance;

constexpr int partialorder(const struct dirent **a, const struct dirent **b) {
  char char_a = (*a)->d_name[0];
  char char_b = (*b)->d_name[0];

  auto get_weight = [](char c) -> int {
    if (c == 'a')
      return 1;
    if (c == 'b')
      return 2;
    return 3;
  };

  return get_weight(char_a) - get_weight(char_b);
}

constexpr int partialorder_reverse(const struct dirent **a,
                                   const struct dirent **b) {
  return -partialorder(a, b);
}

int omegasort(const struct dirent **a, const struct dirent **b) {
  return -LIBC_NAMESPACE::alphasort(a, b);
}

constexpr int skip_hidden(const struct dirent *entry) {
  return entry->d_name[0] != '.';
}

constexpr int
find_entry_index(LIBC_NAMESPACE::cpp::span<struct dirent *const> list,
                 LIBC_NAMESPACE::cpp::string_view target) {
  int i = 0;
  for (const struct dirent *entry : list) {
    if (entry != nullptr &&
        LIBC_NAMESPACE::cpp::string_view(entry->d_name) == target)
      return i;
    ++i;
  }
  return -1;
}

} // anonymous namespace

class LlvmLibcScanImplTest : public LIBC_NAMESPACE::testing::Test {
protected:
  LIBC_NAMESPACE::DirScanResult result{};

  void SetUp() override {
    LIBC_NAMESPACE::testing::Test::SetUp();
    g_mock_config = MockDirTestConfig{};
    result = LIBC_NAMESPACE::DirScanResult{};
  }

  void TearDown() override {
    if (result.entries != nullptr) {
      for (struct dirent *entry : entries())
        delete[] reinterpret_cast<uint8_t *>(entry);
      delete[] result.entries;
      result = LIBC_NAMESPACE::DirScanResult{};
    }
    LIBC_NAMESPACE::testing::Test::TearDown();
  }

  LIBC_NAMESPACE::cpp::span<struct dirent *> entries() const {
    return {result.entries, result.count};
  }

  LIBC_NAMESPACE::ErrorOr<LIBC_NAMESPACE::DirScanResult>
  scan(const char *path = "fake/path",
       int (*filter)(const struct dirent *) = nullptr,
       int (*compare)(const struct dirent **,
                      const struct dirent **) = nullptr) {
    if (result.entries != nullptr)
      TearDown();
    auto res =
        LIBC_NAMESPACE::internal::scan_impl<MockDir>(path, filter, compare);
    if (res.has_value())
      result = res.value();
    else
      result = LIBC_NAMESPACE::DirScanResult{};
    return res;
  }
};

TEST_F(LlvmLibcScanImplTest, SuccessfulRun) {
  auto res = scan();
  ASSERT_TRUE(res.has_value()) << "scan_impl failed: " << res.error();
  ASSERT_EQ(res.value().count, TEST_FILES_COUNT);
  ASSERT_NE(result.entries, nullptr);
  EXPECT_TRUE(g_mock_config.was_closed);

  size_t i = 0;
  for (const struct dirent *entry : entries()) {
    ASSERT_NE(entry, nullptr);
    EXPECT_STREQ(entry->d_name, TEST_FILES[i++]);
  }
}

TEST_F(LlvmLibcScanImplTest, OpenFails) {
  g_mock_config.open_errno_val = EACCES;

  auto res = scan();
  ASSERT_FALSE(res.has_value())
      << "Expected failure, but succeeded with count: " << res.value().count;
  EXPECT_EQ(res.error(), EACCES);
  EXPECT_EQ(result.entries, nullptr);
  EXPECT_FALSE(g_mock_config.was_closed);
}

TEST_F(LlvmLibcScanImplTest, ReadFailsAtStart) {
  g_mock_config.read_errno_val = ENOENT;
  g_mock_config.read_fails_at = 0;

  auto res = scan();
  ASSERT_FALSE(res.has_value())
      << "Expected failure, but succeeded with count: " << res.value().count;
  EXPECT_EQ(res.error(), ENOENT);
  EXPECT_EQ(result.entries, nullptr);
  EXPECT_TRUE(g_mock_config.was_closed);
}

TEST_F(LlvmLibcScanImplTest, ReadFailsMidway) {
  g_mock_config.read_errno_val = ENOENT;
  g_mock_config.read_fails_at = 3;

  auto res = scan();
  ASSERT_FALSE(res.has_value())
      << "Expected failure, but succeeded with count: " << res.value().count;
  EXPECT_EQ(res.error(), ENOENT);
  EXPECT_EQ(result.entries, nullptr);
  EXPECT_TRUE(g_mock_config.was_closed);
}

TEST_F(LlvmLibcScanImplTest, TestPartialOrdering) {
  auto res = scan("fake/path", nullptr, partialorder);
  ASSERT_TRUE(res.has_value()) << "scan_impl failed: " << res.error();
  ASSERT_EQ(res.value().count, TEST_FILES_COUNT);
  ASSERT_NE(result.entries, nullptr);
  EXPECT_TRUE(g_mock_config.was_closed);

  int a_index = find_entry_index(entries(), "a.md");
  int b_index = find_entry_index(entries(), "b.txt");
  ASSERT_NE(a_index, -1);
  ASSERT_NE(b_index, -1);
  EXPECT_LT(a_index, b_index);
}

TEST_F(LlvmLibcScanImplTest, TestPartialOrderingReverse) {
  auto res = scan("fake/path", nullptr, partialorder_reverse);
  ASSERT_TRUE(res.has_value()) << "scan_impl failed: " << res.error();
  ASSERT_EQ(res.value().count, TEST_FILES_COUNT);
  ASSERT_NE(result.entries, nullptr);
  EXPECT_TRUE(g_mock_config.was_closed);

  int a_index = find_entry_index(entries(), "a.md");
  int b_index = find_entry_index(entries(), "b.txt");
  ASSERT_NE(a_index, -1);
  ASSERT_NE(b_index, -1);
  EXPECT_GT(a_index, b_index);
}

TEST_F(LlvmLibcScanImplTest, TestTotalOrderingAZ) {
  auto res = scan("fake/path", nullptr, LIBC_NAMESPACE::alphasort);
  ASSERT_TRUE(res.has_value()) << "scan_impl failed: " << res.error();
  ASSERT_EQ(res.value().count, TEST_FILES_COUNT);
  ASSERT_NE(result.entries, nullptr);
  EXPECT_TRUE(g_mock_config.was_closed);

  constexpr const char *DESIRED_ORDER[] = {".", "..", "a.md", "b.txt", "c.pdf"};
  size_t i = 0;
  for (const struct dirent *entry : entries()) {
    ASSERT_NE(entry, nullptr);
    EXPECT_STREQ(entry->d_name, DESIRED_ORDER[i++]);
  }
}

TEST_F(LlvmLibcScanImplTest, TestTotalOrderingZA) {
  auto res = scan("fake/path", nullptr, omegasort);
  ASSERT_TRUE(res.has_value()) << "scan_impl failed: " << res.error();
  ASSERT_EQ(res.value().count, TEST_FILES_COUNT);
  ASSERT_NE(result.entries, nullptr);
  EXPECT_TRUE(g_mock_config.was_closed);

  constexpr const char *DESIRED_ORDER[] = {"c.pdf", "b.txt", "a.md", "..", "."};
  size_t i = 0;
  for (const struct dirent *entry : entries()) {
    ASSERT_NE(entry, nullptr);
    EXPECT_STREQ(entry->d_name, DESIRED_ORDER[i++]);
  }
}

TEST_F(LlvmLibcScanImplTest, TestFilter) {
  auto res = scan("fake/path", skip_hidden, nullptr);
  ASSERT_TRUE(res.has_value()) << "scan_impl failed: " << res.error();
  constexpr size_t DESIRED_COUNT = TEST_FILES_COUNT - 2;
  ASSERT_EQ(res.value().count, DESIRED_COUNT);
  ASSERT_NE(result.entries, nullptr);
  EXPECT_TRUE(g_mock_config.was_closed);

  constexpr const char *DESIRED_FILES[] = {"b.txt", "a.md", "c.pdf"};
  size_t i = 0;
  for (const struct dirent *entry : entries()) {
    ASSERT_NE(entry, nullptr);
    EXPECT_STREQ(entry->d_name, DESIRED_FILES[i++]);
  }
}

TEST_F(LlvmLibcScanImplTest, EmptyDirectory) {
  g_mock_config.files_count = 0;

  auto res = scan();
  ASSERT_TRUE(res.has_value()) << "scan_impl failed: " << res.error();
  EXPECT_EQ(res.value().count, 0UL);
  EXPECT_EQ(result.entries, nullptr);
  EXPECT_TRUE(g_mock_config.was_closed);
}
