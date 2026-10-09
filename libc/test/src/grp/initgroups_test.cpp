//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for initgroups.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_group.h"
#include "src/grp/endgrent.h"
#include "src/grp/getgrent.h"
#include "src/grp/grp_utils.h"
#include "src/grp/initgroups.h"
#include "src/grp/setgrent.h"
#include "src/unistd/getuid.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/grp/grp_test_utils.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST_F(LlvmLibcGrpTest, InitgroupsPrivilegeCheck) {
  const char *content = "root:x:0:root\n"
                        "users:x:100:user1,user2\n"
                        "admins:x:300:user1\n";
  ScopedGroupFile test_file(libc_make_test_file_path("initgroups_priv.test"),
                            content);

  if (LIBC_NAMESPACE::getuid() == 0) {
    EXPECT_THAT(LIBC_NAMESPACE::initgroups("user1", 1000), Succeeds(0));
  } else {
    EXPECT_THAT(LIBC_NAMESPACE::initgroups("user1", 1000), Fails(EPERM));
  }
}

TEST_F(LlvmLibcGrpTest, InitgroupsNonexistentUser) {
  const char *content = "wheel:x:10:root\n";
  ScopedGroupFile test_file(
      libc_make_test_file_path("initgroups_nonexist.test"), content);

  if (LIBC_NAMESPACE::getuid() == 0) {
    EXPECT_THAT(LIBC_NAMESPACE::initgroups("nonexistent", 1000), Succeeds(0));
  } else {
    EXPECT_THAT(LIBC_NAMESPACE::initgroups("nonexistent", 1000), Fails(EPERM));
  }
}

TEST_F(LlvmLibcGrpTest, InitgroupsNonexistentFile) {
  LIBC_NAMESPACE::grp::TESTONLY_set_group_path(
      libc_make_test_file_path("initgroups_missing.test"));

  if (LIBC_NAMESPACE::getuid() == 0) {
    EXPECT_THAT(LIBC_NAMESPACE::initgroups("anyuser", 1000), Succeeds(0));
  } else {
    EXPECT_THAT(LIBC_NAMESPACE::initgroups("anyuser", 1000), Fails(EPERM));
  }
}

TEST_F(LlvmLibcGrpTest, InitgroupsDoesNotDisturbIteration) {
  const char *content = "group1:x:1:user1\n"
                        "group2:x:2:user2\n"
                        "group3:x:3:user1\n";
  ScopedGroupFile test_file(libc_make_test_file_path("initgroups_iter.test"),
                            content);

  LIBC_NAMESPACE::setgrent();
  const auto first = LIBC_NAMESPACE::grp::read_next();
  ASSERT_TRUE(first.has_value());
  ASSERT_NE(first.value(), nullptr);
  EXPECT_STREQ(first.value()->gr_name, "group1");

  // initgroups reads the database using a scoped stream.
  if (LIBC_NAMESPACE::getuid() == 0) {
    EXPECT_THAT(LIBC_NAMESPACE::initgroups("user2", 50), Succeeds(0));
  } else {
    EXPECT_THAT(LIBC_NAMESPACE::initgroups("user2", 50), Fails(EPERM));
  }

  const auto second = LIBC_NAMESPACE::grp::read_next();
  ASSERT_TRUE(second.has_value());
  ASSERT_NE(second.value(), nullptr);
  EXPECT_STREQ(second.value()->gr_name, "group2");

  LIBC_NAMESPACE::endgrent();
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST_F(LlvmLibcGrpTest, NullUserCrash) {
  ASSERT_DEATH([] { LIBC_NAMESPACE::initgroups(nullptr, 1000); },
               WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
