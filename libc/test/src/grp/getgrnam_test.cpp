//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for getgrnam.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_group.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/libc_errno.h"
#include "src/grp/endgrent.h"
#include "src/grp/getgrent.h"
#include "src/grp/getgrnam.h"
#include "src/grp/grp_utils.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/grp/grp_test_utils.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LlvmLibcGetgrnamTest = LlvmLibcGrpTest;

TEST_F(LlvmLibcGetgrnamTest, Success) {
  const char *content = "root:x:0:\n"
                        "wheel:x:10:alice,bob\n"
                        "users:x:100:carol\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrnam_success.test"),
                            content);

  struct group *grp = LIBC_NAMESPACE::getgrnam("wheel");
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "wheel");
  EXPECT_STREQ(grp->gr_passwd, "x");
  EXPECT_EQ(grp->gr_gid, static_cast<gid_t>(10));
  ASSERT_NE(grp->gr_mem, nullptr);
  EXPECT_STREQ(grp->gr_mem[0], "alice");
  EXPECT_STREQ(grp->gr_mem[1], "bob");
  EXPECT_EQ(grp->gr_mem[2], nullptr);
}

TEST_F(LlvmLibcGetgrnamTest, FirstAndLastEntries) {
  const char *content = "first:x:100:\n"
                        "middle:x:101:\n"
                        "last:x:102:zed\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrnam_boundary.test"),
                            content);

  struct group *grp = LIBC_NAMESPACE::getgrnam("first");
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "first");
  EXPECT_EQ(grp->gr_gid, static_cast<gid_t>(100));
  ASSERT_NE(grp->gr_mem, nullptr);
  EXPECT_EQ(grp->gr_mem[0], nullptr);

  grp = LIBC_NAMESPACE::getgrnam("last");
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "last");
  EXPECT_EQ(grp->gr_gid, static_cast<gid_t>(102));
  ASSERT_NE(grp->gr_mem, nullptr);
  EXPECT_STREQ(grp->gr_mem[0], "zed");
  EXPECT_EQ(grp->gr_mem[1], nullptr);
}

TEST_F(LlvmLibcGetgrnamTest, NotFound) {
  const char *content = "root:x:0:\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrnam_notfound.test"),
                            content);

  // POSIX specifies that errno must not be changed when an entry is not found.
  // Pre-set errno to confirm it remains untouched across the call.
  LIBC_NAMESPACE::libc_errno = ENOENT;
  struct group *grp = LIBC_NAMESPACE::getgrnam("nonexistent");
  EXPECT_EQ(grp, nullptr);
  ASSERT_ERRNO_EQ(ENOENT);

  // When errno is initially zero, it remains zero.
  LIBC_NAMESPACE::libc_errno = 0;
  grp = LIBC_NAMESPACE::getgrnam("nonexistent");
  EXPECT_EQ(grp, nullptr);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcGetgrnamTest, BlankLines) {
  const char *content = "\nroot:x:0:\n\n\n"
                        "wheel:x:10:alice\n\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrnam_blank.test"),
                            content);

  struct group *grp = LIBC_NAMESPACE::getgrnam("wheel");
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "wheel");
  EXPECT_EQ(grp->gr_gid, static_cast<gid_t>(10));
}

TEST_F(LlvmLibcGetgrnamTest, FileOpenFailure) {
  const auto missing_path =
      libc_make_test_file_path("nonexistent_dir/getgrnam_missing.test");
  LIBC_NAMESPACE::grp::TESTONLY_set_group_path(missing_path);

  ASSERT_THAT(reinterpret_cast<void *>(LIBC_NAMESPACE::getgrnam("root")),
              Fails(ENOENT, static_cast<void *>(nullptr)));
}

TEST_F(LlvmLibcGetgrnamTest, LongRecordGrowsBuffer) {
  // 600 four-character member names plus commas (~3,000 bytes) requires
  // dynamic buffer growth for both the line and its member pointer array.
  constexpr size_t MEMBER_COUNT = 600;
  constexpr size_t FILE_BUFFER_SIZE = 8192;

  char content[FILE_BUFFER_SIZE];
  size_t offset = 0;
  constexpr char HEADER[] = "small:x:1:\nbiggroup:x:2000:";
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

  ScopedGroupFile test_file(libc_make_test_file_path("getgrnam_long.test"),
                            content);

  struct group *grp = LIBC_NAMESPACE::getgrnam("biggroup");
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "biggroup");
  EXPECT_EQ(grp->gr_gid, static_cast<gid_t>(2000));
  ASSERT_NE(grp->gr_mem, nullptr);
  EXPECT_STREQ(grp->gr_mem[0], "u000");
  EXPECT_STREQ(grp->gr_mem[MEMBER_COUNT / 2], "u300");
  EXPECT_STREQ(grp->gr_mem[MEMBER_COUNT - 1], "u599");
  EXPECT_EQ(grp->gr_mem[MEMBER_COUNT], nullptr);

  // The buffer has grown; a short record afterwards still works.
  grp = LIBC_NAMESPACE::getgrnam("small");
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "small");
}

TEST_F(LlvmLibcGetgrnamTest, DoesNotDisturbIteration) {
  const char *content = "root:x:0:\n"
                        "wheel:x:10:alice\n"
                        "users:x:100:carol\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrnam_iter.test"),
                            content);

  struct group *grp = LIBC_NAMESPACE::getgrent();
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "root");

  struct group *found = LIBC_NAMESPACE::getgrnam("users");
  ASSERT_NE(found, nullptr);
  EXPECT_STREQ(found->gr_name, "users");

  // Closing the iteration stream via endgrent does not invalidate the pointer
  // returned by getgrnam.
  LIBC_NAMESPACE::endgrent();
  EXPECT_STREQ(found->gr_name, "users");

  // Reopening iteration starts from the beginning.
  grp = LIBC_NAMESPACE::getgrent();
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "root");

  // A mid-iteration getgrnam lookup opens its own scoped stream so the
  // getgrent stream position is undisturbed.
  found = LIBC_NAMESPACE::getgrnam("users");
  ASSERT_NE(found, nullptr);
  EXPECT_STREQ(found->gr_name, "users");

  grp = LIBC_NAMESPACE::getgrent();
  ASSERT_NE(grp, nullptr);
  EXPECT_STREQ(grp->gr_name, "wheel");

  LIBC_NAMESPACE::endgrent();
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST_F(LlvmLibcGetgrnamTest, NullPointerCrash) {
  ASSERT_DEATH([] { LIBC_NAMESPACE::getgrnam(nullptr); }, WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
