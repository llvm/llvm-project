//===-- Unittests for Linux sysinfo support -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/linux/sysinfo.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "src/unistd/write.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

static int write_test_file(const char *path, cpp::string_view contents) {
  int fd = LIBC_NAMESPACE::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0)
    return fd;

  if (LIBC_NAMESPACE::write(fd, contents.data(), contents.size()) !=
      static_cast<ssize_t>(contents.size())) {
    LIBC_NAMESPACE::close(fd);
    return -1;
  }

  return LIBC_NAMESPACE::close(fd);
}

TEST(LlvmLibcOSUtilSysinfoTest, PossibleCpuCountSmokeTest) {
  cpp::optional<size_t> parsed =
      sysinfo::parse_nproc_from(sysinfo::POSSIBLE_NPROC_PATH);
  ASSERT_TRUE(static_cast<bool>(parsed));
  EXPECT_GT(*parsed, size_t(0));
  EXPECT_GT(
      sysinfo::parse_nproc_with_fallback_from(sysinfo::POSSIBLE_NPROC_PATH),
      size_t(0));
}

TEST(LlvmLibcOSUtilSysinfoTest, OnlineCpuCountSmokeTest) {
  cpp::optional<size_t> parsed =
      sysinfo::parse_nproc_from(sysinfo::ONLINE_NPROC_PATH);
  ASSERT_TRUE(static_cast<bool>(parsed));
  EXPECT_GT(*parsed, size_t(0));
  EXPECT_GT(sysinfo::parse_nproc_with_fallback_from(sysinfo::ONLINE_NPROC_PATH),
            size_t(0));
}

TEST(LlvmLibcOSUtilSysinfoTest, SyntheticCpuLists) {
  constexpr char FILENAME[] =
      APPEND_LIBC_TEST("sysinfo.synthetic_cpu_list.test");
  CString test_file = libc_make_test_file_path(FILENAME);

  struct TestCase {
    cpp::string_view contents;
    cpp::optional<size_t> expected;
  };

  constexpr TestCase TEST_CASES[] = {
      {"0\n", 1},
      {"0-7\n", 8},
      {"0-0,2,4-6\n", 5},
      {"0-3,8-11\n", 8},
      {"0-3,8-11,16\n", 9},
      {"1,2,3,4-9,99\n", 10},
      {"3-1\n", cpp::nullopt},
      {"0-\n", cpp::nullopt},
  };

  for (const TestCase &test_case : TEST_CASES) {
    ASSERT_EQ(write_test_file(test_file, test_case.contents), 0);
    cpp::optional<size_t> parsed = sysinfo::parse_nproc_from(test_file);
    EXPECT_EQ(static_cast<bool>(parsed), static_cast<bool>(test_case.expected));
    if (parsed)
      EXPECT_EQ(*parsed, *test_case.expected);
    EXPECT_GT(sysinfo::parse_nproc_with_fallback_from(test_file), size_t(0));
  }

  ASSERT_EQ(LIBC_NAMESPACE::unlink(test_file), 0);
}

TEST(LlvmLibcOSUtilSysinfoTest, NonexistentPath) {
  constexpr const char *TEST_PATH =
      "/not-exist-at-all-path-for-libc-nproc-test";

  EXPECT_FALSE(static_cast<bool>(sysinfo::parse_nproc_from(TEST_PATH)));
  EXPECT_GT(sysinfo::parse_nproc_with_fallback_from(TEST_PATH), size_t(0));
}

} // namespace LIBC_NAMESPACE_DECL
