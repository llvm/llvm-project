//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for getgrouplist.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_group.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/stringstream.h"
#include "src/grp/endgrent.h"
#include "src/grp/getgrent.h"
#include "src/grp/getgrouplist.h"
#include "src/grp/grp_utils.h"
#include "src/grp/setgrent.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/grp/grp_test_utils.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST_F(LlvmLibcGrpTest, GetgrouplistSuccess) {
  const char *content = "root:x:0:root\n"
                        "bin:x:1:bin,daemon\n"
                        "users:x:100:user1,user2\n"
                        "developers:x:200:user2,user3\n"
                        "admins:x:300:user1,user3\n";
  ScopedGroupFile test_file(
      libc_make_test_file_path("getgrouplist_success.test"), content);

  gid_t groups[10];
  int ngroups = 10;

  ASSERT_THAT(LIBC_NAMESPACE::getgrouplist("user1", 1000, groups, &ngroups),
              Succeeds(3));
  EXPECT_EQ(ngroups, 3);
  EXPECT_EQ(groups[0], static_cast<gid_t>(1000));
  EXPECT_EQ(groups[1], static_cast<gid_t>(100));
  EXPECT_EQ(groups[2], static_cast<gid_t>(300));
}

TEST_F(LlvmLibcGrpTest, BaseGroupAlreadyInDatabase) {
  const char *content = "staff:x:50:user1\n"
                        "devel:x:60:user1\n";
  ScopedGroupFile test_file(
      libc_make_test_file_path("getgrouplist_base_dup.test"), content);

  gid_t groups[10];
  int ngroups = 10;

  ASSERT_THAT(LIBC_NAMESPACE::getgrouplist("user1", 50, groups, &ngroups),
              Succeeds(2));
  EXPECT_EQ(ngroups, 2);
  EXPECT_EQ(groups[0], static_cast<gid_t>(50));
  EXPECT_EQ(groups[1], static_cast<gid_t>(60));
}

TEST_F(LlvmLibcGrpTest, DuplicateGidInDatabase) {
  const char *content = "team1:x:100:user1\n"
                        "team2:x:100:user1\n"
                        "team3:x:200:user1\n";
  ScopedGroupFile test_file(
      libc_make_test_file_path("getgrouplist_dup_gid.test"), content);

  gid_t groups[10];
  int ngroups = 10;

  ASSERT_THAT(LIBC_NAMESPACE::getgrouplist("user1", 500, groups, &ngroups),
              Succeeds(3));
  EXPECT_EQ(ngroups, 3);
  EXPECT_EQ(groups[0], static_cast<gid_t>(500));
  EXPECT_EQ(groups[1], static_cast<gid_t>(100));
  EXPECT_EQ(groups[2], static_cast<gid_t>(200));
}

TEST_F(LlvmLibcGrpTest, BufferTooSmall) {
  const char *content = "users:x:100:user1\n"
                        "admins:x:200:user1\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrouplist_small.test"),
                            content);

  gid_t groups[1] = {0};
  int ngroups = 1;

  EXPECT_EQ(LIBC_NAMESPACE::getgrouplist("user1", 10, groups, &ngroups), -1);
  EXPECT_EQ(ngroups, 3);
  EXPECT_EQ(groups[0], static_cast<gid_t>(10));
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcGrpTest, ZeroBufferQuery) {
  const char *content = "users:x:100:user1\n"
                        "admins:x:200:user1\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrouplist_zero.test"),
                            content);

  int ngroups = 0;

  EXPECT_EQ(LIBC_NAMESPACE::getgrouplist("user1", 10, nullptr, &ngroups), -1);
  EXPECT_EQ(ngroups, 3);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcGrpTest, UserNotFound) {
  const char *content = "wheel:x:10:root\n";
  ScopedGroupFile test_file(
      libc_make_test_file_path("getgrouplist_not_found.test"), content);

  gid_t groups[5];
  int ngroups = 5;

  ASSERT_THAT(
      LIBC_NAMESPACE::getgrouplist("nonexistent", 1000, groups, &ngroups),
      Succeeds(1));
  EXPECT_EQ(ngroups, 1);
  EXPECT_EQ(groups[0], static_cast<gid_t>(1000));
}

TEST_F(LlvmLibcGrpTest, NonexistentFile) {
  LIBC_NAMESPACE::grp::TESTONLY_set_group_path(
      libc_make_test_file_path("getgrouplist_missing.test"));

  gid_t groups[5];
  int ngroups = 5;

  ASSERT_THAT(LIBC_NAMESPACE::getgrouplist("anyuser", 1000, groups, &ngroups),
              Succeeds(1));
  EXPECT_EQ(ngroups, 1);
  EXPECT_EQ(groups[0], static_cast<gid_t>(1000));
}

TEST_F(LlvmLibcGrpTest, DoesNotDisturbIteration) {
  const char *content = "group1:x:1:user1\n"
                        "group2:x:2:user2\n"
                        "group3:x:3:user1\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrouplist_iter.test"),
                            content);

  LIBC_NAMESPACE::setgrent();
  const auto first = LIBC_NAMESPACE::grp::read_next();
  ASSERT_TRUE(first.has_value());
  ASSERT_NE(first.value(), nullptr);
  EXPECT_STREQ(first.value()->gr_name, "group1");

  gid_t groups[5];
  int ngroups = 5;
  ASSERT_THAT(LIBC_NAMESPACE::getgrouplist("user2", 50, groups, &ngroups),
              Succeeds(2));
  EXPECT_EQ(ngroups, 2);

  const auto second = LIBC_NAMESPACE::grp::read_next();
  ASSERT_TRUE(second.has_value());
  ASSERT_NE(second.value(), nullptr);
  EXPECT_STREQ(second.value()->gr_name, "group2");

  LIBC_NAMESPACE::endgrent();
}

TEST_F(LlvmLibcGrpTest, UserWithMoreThan32Groups) {
  // Construct a group file containing 35 groups with testuser.
  // This exercises dynamic reallocation in GidList.
  LIBC_NAMESPACE::cpp::array<char, 2048> content;
  LIBC_NAMESPACE::cpp::StringStream stream(content);
  for (int i = 1; i <= 35; ++i)
    stream << "grp" << i << ":x:" << (1000 + i) << ":testuser\n";
  stream << LIBC_NAMESPACE::cpp::StringStream::ENDS;
  ASSERT_FALSE(stream.overflow());

  ScopedGroupFile test_file(libc_make_test_file_path("getgrouplist_many.test"),
                            content.data());

  gid_t groups[64];
  int ngroups = 64;

  ASSERT_THAT(LIBC_NAMESPACE::getgrouplist("testuser", 5000, groups, &ngroups),
              Succeeds(36));
  EXPECT_EQ(ngroups, 36);
  EXPECT_EQ(groups[0], static_cast<gid_t>(5000));
  for (int i = 1; i <= 35; ++i)
    EXPECT_EQ(groups[i], static_cast<gid_t>(1000 + i));
}

TEST_F(LlvmLibcGrpTest, NegativeNgroupsReturnsEinval) {
  int ngroups = -1;
  gid_t groups[5];
  EXPECT_THAT(LIBC_NAMESPACE::getgrouplist("user", 100, groups, &ngroups),
              Fails(EINVAL));
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST_F(LlvmLibcGrpTest, NullPointerCrash) {
  int ngroups = 5;
  gid_t groups[5];

  ASSERT_DEATH(
      [&] { LIBC_NAMESPACE::getgrouplist(nullptr, 100, groups, &ngroups); },
      WITH_SIGNAL(-1));
  ASSERT_DEATH(
      [&] { LIBC_NAMESPACE::getgrouplist("user", 100, groups, nullptr); },
      WITH_SIGNAL(-1));
  ASSERT_DEATH(
      [&] { LIBC_NAMESPACE::getgrouplist("user", 100, nullptr, &ngroups); },
      WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
