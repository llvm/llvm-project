//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for getgrent, setgrent, and endgrent.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_group.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/libc_errno.h"
#include "src/grp/endgrent.h"
#include "src/grp/getgrent.h"
#include "src/grp/grp_utils.h"
#include "src/grp/setgrent.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/grp/grp_test_utils.h"

using LIBC_NAMESPACE::libc_errno;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;

TEST_F(LlvmLibcGrpTest, GetGrentTestSuccess) {
  constexpr char CONTENT[] = "root:x:0:root\n"
                             "bin:x:1:bin,daemon\n"
                             "wheel:x:10:root,admin,user1\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrent_success.test"),
                            CONTENT);

  struct group *grp = LIBC_NAMESPACE::getgrent();
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "root");
  EXPECT_STREQ(grp->gr_passwd, "x");
  EXPECT_EQ(grp->gr_gid, static_cast<gid_t>(0));
  ASSERT_NE(grp->gr_mem, nullptr);
  EXPECT_STREQ(grp->gr_mem[0], "root");
  EXPECT_EQ(grp->gr_mem[1], nullptr);

  grp = LIBC_NAMESPACE::getgrent();
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "bin");
  EXPECT_STREQ(grp->gr_passwd, "x");
  EXPECT_EQ(grp->gr_gid, static_cast<gid_t>(1));
  ASSERT_NE(grp->gr_mem, nullptr);
  EXPECT_STREQ(grp->gr_mem[0], "bin");
  EXPECT_STREQ(grp->gr_mem[1], "daemon");
  EXPECT_EQ(grp->gr_mem[2], nullptr);

  grp = LIBC_NAMESPACE::getgrent();
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "wheel");
  EXPECT_STREQ(grp->gr_passwd, "x");
  EXPECT_EQ(grp->gr_gid, static_cast<gid_t>(10));
  ASSERT_NE(grp->gr_mem, nullptr);
  EXPECT_STREQ(grp->gr_mem[0], "root");
  EXPECT_STREQ(grp->gr_mem[1], "admin");
  EXPECT_STREQ(grp->gr_mem[2], "user1");
  EXPECT_EQ(grp->gr_mem[3], nullptr);

  // At EOF, getgrent returns nullptr without altering errno.
  libc_errno = 0;
  grp = LIBC_NAMESPACE::getgrent();
  EXPECT_EQ(grp, nullptr);
  ASSERT_ERRNO_SUCCESS();

  LIBC_NAMESPACE::endgrent();
}

TEST_F(LlvmLibcGrpTest, GetGrentTestFailure) {
  constexpr char CONTENT[] = "malformed_line_without_colons\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrent_failure.test"),
                            CONTENT);

  ASSERT_THAT(reinterpret_cast<void *>(LIBC_NAMESPACE::getgrent()),
              Fails(EINVAL, static_cast<void *>(nullptr)));

  LIBC_NAMESPACE::endgrent();
}

TEST_F(LlvmLibcGrpTest, SetGrentTestHermetic) {
  constexpr char CONTENT[] = "root:x:0:root\n"
                             "bin:x:1:bin\n";
  ScopedGroupFile test_file(libc_make_test_file_path("setgrent_hermetic.test"),
                            CONTENT);

  struct group *grp = LIBC_NAMESPACE::getgrent();
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "root");

  // Rewind the group database stream to the beginning.
  LIBC_NAMESPACE::setgrent();

  grp = LIBC_NAMESPACE::getgrent();
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "root");

  LIBC_NAMESPACE::endgrent();
}

TEST_F(LlvmLibcGrpTest, ReopenAfterEndgrent) {
  constexpr char CONTENT[] = "root:x:0:root\n"
                             "bin:x:1:bin\n";
  ScopedGroupFile test_file(libc_make_test_file_path("reopen_after_end.test"),
                            CONTENT);

  struct group *grp = LIBC_NAMESPACE::getgrent();
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "root");

  LIBC_NAMESPACE::endgrent();

  // endgrent closes the stream without invalidating the last returned pointer.
  EXPECT_STREQ(grp->gr_name, "root");

  // Subsequent getgrent after endgrent reopens the database from the start.
  grp = LIBC_NAMESPACE::getgrent();
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "root");

  LIBC_NAMESPACE::endgrent();
}

TEST_F(LlvmLibcGrpTest, FileOpenFailure) {
  auto nonexistent_path = libc_make_test_file_path("getgrent_nonexistent.test");
  LIBC_NAMESPACE::grp::TESTONLY_set_group_path(nonexistent_path);

  struct group *grp = LIBC_NAMESPACE::getgrent();
  EXPECT_EQ(grp, nullptr);
  ASSERT_ERRNO_EQ(ENOENT);

  // POSIX specifies that setgrent() sets errno on error.
  libc_errno = 0;
  LIBC_NAMESPACE::setgrent();
  ASSERT_ERRNO_EQ(ENOENT);
}

TEST_F(LlvmLibcGrpTest, BlankLines) {
  constexpr char CONTENT[] = "\n"
                             "root:x:0:root\n"
                             "\n"
                             "\n"
                             "bin:x:1:bin\n"
                             "\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrent_blank.test"),
                            CONTENT);

  struct group *grp = LIBC_NAMESPACE::getgrent();
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "root");

  grp = LIBC_NAMESPACE::getgrent();
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "bin");

  // POSIX mandates that getgrent shall return null and not change errno on EOF.
  libc_errno = ENOENT;
  grp = LIBC_NAMESPACE::getgrent();
  EXPECT_EQ(grp, nullptr);
  ASSERT_ERRNO_EQ(ENOENT);

  LIBC_NAMESPACE::endgrent();
}

TEST_F(LlvmLibcGrpTest, DynamicMemberAndLineGrowth) {
  // Construct a record with 600 members (~3 KB), exercising both dynamic line
  // growth and per-record sizing of the gr_mem pointer array.
  constexpr size_t MEMBER_COUNT = 600;
  constexpr size_t FILE_BUFFER_SIZE = 8192;
  char content[FILE_BUFFER_SIZE];
  size_t offset = 0;
  constexpr char HEADER[] = "biggroup:x:2000:";
  for (size_t i = 0; HEADER[i] != '\0'; ++i)
    content[offset++] = HEADER[i];

  for (size_t i = 0; i < MEMBER_COUNT; ++i) {
    if (i > 0)
      content[offset++] = ',';
    content[offset++] = 'u';
    content[offset++] =
        LIBC_NAMESPACE::internal::int_to_b36_char((i / 100) % 10);
    content[offset++] =
        LIBC_NAMESPACE::internal::int_to_b36_char((i / 10) % 10);
    content[offset++] = LIBC_NAMESPACE::internal::int_to_b36_char(i % 10);
  }
  content[offset++] = '\n';
  content[offset] = '\0';

  ScopedGroupFile test_file(
      libc_make_test_file_path("getgrent_dynamic_growth.test"), content);

  struct group *grp = LIBC_NAMESPACE::getgrent();
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "biggroup");
  EXPECT_STREQ(grp->gr_passwd, "x");
  EXPECT_EQ(grp->gr_gid, static_cast<gid_t>(2000));
  ASSERT_NE(grp->gr_mem, nullptr);

  EXPECT_STREQ(grp->gr_mem[0], "u000");
  EXPECT_STREQ(grp->gr_mem[MEMBER_COUNT / 2], "u300");
  EXPECT_STREQ(grp->gr_mem[MEMBER_COUNT - 1], "u599");
  EXPECT_EQ(grp->gr_mem[MEMBER_COUNT], nullptr);

  LIBC_NAMESPACE::endgrent();
}
