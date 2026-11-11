//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for getgrgid_r.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_group.h"
#include "src/grp/getgrgid_r.h"
#include "src/grp/grp_utils.h"
#include "test/UnitTest/Test.h"
#include "test/src/grp/grp_test_utils.h"

TEST_F(LlvmLibcGrpTest, GetGrgidRSuccess) {
  const char *content = "root:x:0:root\n"
                        "wheel:x:10:root,admin\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrgid_r_success.test"),
                            content);

  struct group grp;
  struct group *result = nullptr;
  char buffer[256];

  ASSERT_EQ(
      LIBC_NAMESPACE::getgrgid_r(10, &grp, buffer, sizeof(buffer), &result), 0);
  ASSERT_NE(result, nullptr);
  EXPECT_STREQ(grp.gr_name, "wheel");
  EXPECT_EQ(grp.gr_gid, static_cast<gid_t>(10));
  ASSERT_NE(grp.gr_mem, nullptr);
  EXPECT_STREQ(grp.gr_mem[0], "root");
  EXPECT_STREQ(grp.gr_mem[1], "admin");
  EXPECT_EQ(grp.gr_mem[2], nullptr);
}

TEST_F(LlvmLibcGrpTest, GetGrgidRRootGidZero) {
  const char *content = "root:x:0:root\n"
                        "wheel:x:10:admin\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrgid_r_zero.test"),
                            content);

  struct group grp;
  struct group *result = nullptr;
  char buffer[256];

  ASSERT_EQ(
      LIBC_NAMESPACE::getgrgid_r(0, &grp, buffer, sizeof(buffer), &result), 0);
  ASSERT_NE(result, nullptr);
  EXPECT_STREQ(grp.gr_name, "root");
  EXPECT_EQ(grp.gr_gid, static_cast<gid_t>(0));
}

TEST_F(LlvmLibcGrpTest, GetGrgidREmptyMemberList) {
  const char *content = "nogroup:x:65534:\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrgid_r_empty.test"),
                            content);

  struct group grp;
  struct group *result = nullptr;
  char buffer[256];

  ASSERT_EQ(
      LIBC_NAMESPACE::getgrgid_r(65534, &grp, buffer, sizeof(buffer), &result),
      0);
  ASSERT_NE(result, nullptr);
  EXPECT_STREQ(grp.gr_name, "nogroup");
  ASSERT_NE(grp.gr_mem, nullptr);
  EXPECT_EQ(grp.gr_mem[0], nullptr);
}

TEST_F(LlvmLibcGrpTest, GetGrgidRNotFound) {
  const char *content = "root:x:0:root\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrgid_r_absent.test"),
                            content);

  struct group grp;
  struct group *result = &grp;
  char buffer[256];

  ASSERT_EQ(
      LIBC_NAMESPACE::getgrgid_r(4242, &grp, buffer, sizeof(buffer), &result),
      0);
  ASSERT_EQ(result, nullptr);
}

TEST_F(LlvmLibcGrpTest, GetGrgidRBufferTooSmall) {
  const char *content = "wheel:x:10:root,admin,user1\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrgid_r_small.test"),
                            content);

  struct group grp;
  struct group *result = &grp;
  char buffer[8];

  ASSERT_EQ(
      LIBC_NAMESPACE::getgrgid_r(10, &grp, buffer, sizeof(buffer), &result),
      ERANGE);
  ASSERT_EQ(result, nullptr);
}

TEST_F(LlvmLibcGrpTest, GetGrgidRBlankLines) {
  const char *content = "\nroot:x:0:root\n\n\nwheel:x:10:admin\n\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrgid_r_blank.test"),
                            content);

  struct group grp;
  struct group *result = nullptr;
  char buffer[256];

  ASSERT_EQ(
      LIBC_NAMESPACE::getgrgid_r(10, &grp, buffer, sizeof(buffer), &result), 0);
  ASSERT_NE(result, nullptr);
  EXPECT_STREQ(grp.gr_name, "wheel");
}

TEST_F(LlvmLibcGrpTest, GetGrgidRDoesNotDisturbIteration) {
  const char *content = "root:x:0:root\n"
                        "bin:x:1:bin\n"
                        "wheel:x:10:admin\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrgid_r_iter.test"),
                            content);

  const auto first = LIBC_NAMESPACE::grp::read_next();
  ASSERT_TRUE(first.has_value());
  ASSERT_NE(first.value(), nullptr);
  EXPECT_STREQ(first.value()->gr_name, "root");

  struct group grp;
  struct group *result = nullptr;
  char buffer[256];
  ASSERT_EQ(
      LIBC_NAMESPACE::getgrgid_r(10, &grp, buffer, sizeof(buffer), &result), 0);
  ASSERT_NE(result, nullptr);

  const auto second = LIBC_NAMESPACE::grp::read_next();
  ASSERT_TRUE(second.has_value());
  ASSERT_NE(second.value(), nullptr);
  EXPECT_STREQ(second.value()->gr_name, "bin");

  LIBC_NAMESPACE::grp::close();
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST_F(LlvmLibcGrpTest, NullPointerCrash) {
  struct group grp;
  struct group *result = nullptr;
  char buffer[64];
  ASSERT_DEATH(
      [&] {
        LIBC_NAMESPACE::getgrgid_r(0, nullptr, buffer, sizeof(buffer), &result);
      },
      WITH_SIGNAL(-1));
  ASSERT_DEATH(
      [&] {
        LIBC_NAMESPACE::getgrgid_r(0, &grp, buffer, sizeof(buffer), nullptr);
      },
      WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
