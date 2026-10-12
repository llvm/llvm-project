//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for getgrnam_r.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_group.h"
#include "src/grp/getgrnam_r.h"
#include "src/grp/grp_utils.h"
#include "test/UnitTest/Test.h"
#include "test/src/grp/grp_test_utils.h"

TEST_F(LlvmLibcGrpTest, GetGrnamRSuccess) {
  const char *content = "root:x:0:root\n"
                        "wheel:x:10:root,admin,user1\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrnam_r_success.test"),
                            content);

  struct group grp;
  struct group *result = nullptr;
  char buffer[256];

  ASSERT_EQ(LIBC_NAMESPACE::getgrnam_r("wheel", &grp, buffer, sizeof(buffer),
                                       &result),
            0);
  ASSERT_EQ(result, &grp);
  EXPECT_STREQ(grp.gr_name, "wheel");
  EXPECT_STREQ(grp.gr_passwd, "x");
  EXPECT_EQ(grp.gr_gid, static_cast<gid_t>(10));
  ASSERT_NE(grp.gr_mem, nullptr);
  EXPECT_STREQ(grp.gr_mem[0], "root");
  EXPECT_STREQ(grp.gr_mem[1], "admin");
  EXPECT_STREQ(grp.gr_mem[2], "user1");
  EXPECT_EQ(grp.gr_mem[3], nullptr);
}

TEST_F(LlvmLibcGrpTest, GetGrnamRResultLivesInCallerBuffer) {
  const char *content = "wheel:x:10:root,admin\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrnam_r_buf.test"),
                            content);

  struct group grp;
  struct group *result = nullptr;
  char buffer[256];

  ASSERT_EQ(LIBC_NAMESPACE::getgrnam_r("wheel", &grp, buffer, sizeof(buffer),
                                       &result),
            0);
  ASSERT_NE(result, nullptr);

  // POSIX requires the strings and the member array to be stored in the
  // caller's buffer, not in any library-owned storage.
  char *begin = buffer;
  char *end = buffer + sizeof(buffer);
  EXPECT_GE(grp.gr_name, begin);
  EXPECT_LT(grp.gr_name, end);
  EXPECT_GE(grp.gr_passwd, begin);
  EXPECT_LT(grp.gr_passwd, end);
  EXPECT_GE(reinterpret_cast<char *>(grp.gr_mem), begin);
  EXPECT_LT(reinterpret_cast<char *>(grp.gr_mem), end);
  EXPECT_GE(grp.gr_mem[0], begin);
  EXPECT_LT(grp.gr_mem[0], end);
}

TEST_F(LlvmLibcGrpTest, GetGrnamRNotFound) {
  const char *content = "root:x:0:root\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrnam_r_absent.test"),
                            content);

  struct group grp;
  struct group *result = &grp;
  char buffer[256];

  ASSERT_EQ(LIBC_NAMESPACE::getgrnam_r("nosuchgroup", &grp, buffer,
                                       sizeof(buffer), &result),
            0);
  // Not found is not an error: zero is returned with a null result.
  ASSERT_EQ(result, nullptr);
}

TEST_F(LlvmLibcGrpTest, GetGrnamRBufferTooSmallForRecord) {
  const char *content = "wheel:x:10:root,admin,user1\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrnam_r_small.test"),
                            content);

  struct group grp;
  struct group *result = &grp;
  char buffer[8];

  ASSERT_EQ(LIBC_NAMESPACE::getgrnam_r("wheel", &grp, buffer, sizeof(buffer),
                                       &result),
            ERANGE);
  ASSERT_EQ(result, nullptr);
}

TEST_F(LlvmLibcGrpTest, GetGrnamRBufferTooSmallForMemberArray) {
  // The record fits, but the member pointer array carved out of the space
  // after it does not. This must be reported as ERANGE rather than silently
  // truncating the member list.
  const char *content = "wheel:x:10:a,b,c,d,e,f,g,h\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrnam_r_mem.test"),
                            content);

  struct group grp;
  struct group *result = &grp;
  // "wheel:x:10:a,b,c,d,e,f,g,h" is 26 bytes excluding the newline. Adding 1
  // for the null terminator leaves 5 bytes in a 32-byte buffer, which is too
  // small for the 9 char * pointers needed by 8 members plus the terminating
  // nullptr.
  constexpr size_t RECORD_LEN = 26;
  char buffer[RECORD_LEN + 6];

  ASSERT_EQ(LIBC_NAMESPACE::getgrnam_r("wheel", &grp, buffer, sizeof(buffer),
                                       &result),
            ERANGE);
  ASSERT_EQ(result, nullptr);
}

TEST_F(LlvmLibcGrpTest, GetGrnamRDoesNotDisturbIteration) {
  const char *content = "root:x:0:root\n"
                        "bin:x:1:bin\n"
                        "wheel:x:10:admin\n";
  ScopedGroupFile test_file(libc_make_test_file_path("getgrnam_r_iter.test"),
                            content);

  const auto first = LIBC_NAMESPACE::grp::read_next();
  ASSERT_TRUE(first.has_value());
  ASSERT_NE(first.value(), nullptr);
  EXPECT_STREQ(first.value()->gr_name, "root");

  struct group grp;
  struct group *result = nullptr;
  char buffer[256];
  ASSERT_EQ(LIBC_NAMESPACE::getgrnam_r("wheel", &grp, buffer, sizeof(buffer),
                                       &result),
            0);
  ASSERT_NE(result, nullptr);

  // The reentrant lookup uses its own stream, so iteration continues from
  // where it left off.
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
      [] {
        struct group local_grp;
        struct group *local_result = nullptr;
        char local_buf[64];
        LIBC_NAMESPACE::getgrnam_r(nullptr, &local_grp, local_buf,
                                   sizeof(local_buf), &local_result);
      },
      WITH_SIGNAL(-1));
  ASSERT_DEATH(
      [&] {
        LIBC_NAMESPACE::getgrnam_r("root", nullptr, buffer, sizeof(buffer),
                                   &result);
      },
      WITH_SIGNAL(-1));
  ASSERT_DEATH(
      [&] {
        LIBC_NAMESPACE::getgrnam_r("root", &grp, buffer, sizeof(buffer),
                                   nullptr);
      },
      WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
