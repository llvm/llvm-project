//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for parse_group_line.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_group.h"
#include "src/__support/CPP/span.h"
#include "src/__support/pwd/flat_file_db.h"
#include "src/grp/grp_utils.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcGrpUtilsTest, ParseGroupLine_Success) {
  char line[] = "wheel:x:10:root,admin,user1";
  char *mem_ptrs[8];
  struct group grp;
  const bool ok = LIBC_NAMESPACE::grp::parse_group_line(line, &grp, mem_ptrs);
  ASSERT_TRUE(ok);
  EXPECT_STREQ(grp.gr_name, "wheel");
  EXPECT_STREQ(grp.gr_passwd, "x");
  EXPECT_EQ(grp.gr_gid, static_cast<gid_t>(10));
  ASSERT_NE(grp.gr_mem, nullptr);
  EXPECT_STREQ(grp.gr_mem[0], "root");
  EXPECT_STREQ(grp.gr_mem[1], "admin");
  EXPECT_STREQ(grp.gr_mem[2], "user1");
  EXPECT_EQ(grp.gr_mem[3], nullptr);
}

TEST(LlvmLibcGrpUtilsTest, ParseGroupLine_EmptyMembers) {
  char line[] = "nogroup:x:65534:";
  char *mem_ptrs[4];
  struct group grp;
  const bool ok = LIBC_NAMESPACE::grp::parse_group_line(line, &grp, mem_ptrs);
  ASSERT_TRUE(ok);
  EXPECT_STREQ(grp.gr_name, "nogroup");
  EXPECT_STREQ(grp.gr_passwd, "x");
  EXPECT_EQ(grp.gr_gid, static_cast<gid_t>(65534));
  ASSERT_NE(grp.gr_mem, nullptr);
  EXPECT_EQ(grp.gr_mem[0], nullptr);
}

TEST(LlvmLibcGrpUtilsTest, ParseGroupLine_LeadingTrailingAndConsecutiveCommas) {
  char line[] = "test:x:100:,user1,,user2,";
  char *mem_ptrs[8];
  struct group grp;
  const bool ok = LIBC_NAMESPACE::grp::parse_group_line(line, &grp, mem_ptrs);
  ASSERT_TRUE(ok);
  EXPECT_STREQ(grp.gr_name, "test");
  EXPECT_STREQ(grp.gr_passwd, "x");
  EXPECT_EQ(grp.gr_gid, static_cast<gid_t>(100));
  ASSERT_NE(grp.gr_mem, nullptr);
  EXPECT_STREQ(grp.gr_mem[0], "user1");
  EXPECT_STREQ(grp.gr_mem[1], "user2");
  EXPECT_EQ(grp.gr_mem[2], nullptr);
}

TEST(LlvmLibcGrpUtilsTest, ParseGroupLine_SingleMember) {
  char line[] = "bin:x:1:bin";
  char *mem_ptrs[4];
  struct group grp;
  const bool ok = LIBC_NAMESPACE::grp::parse_group_line(line, &grp, mem_ptrs);
  ASSERT_TRUE(ok);
  EXPECT_STREQ(grp.gr_name, "bin");
  EXPECT_STREQ(grp.gr_passwd, "x");
  EXPECT_EQ(grp.gr_gid, static_cast<gid_t>(1));
  ASSERT_NE(grp.gr_mem, nullptr);
  EXPECT_STREQ(grp.gr_mem[0], "bin");
  EXPECT_EQ(grp.gr_mem[1], nullptr);
}

TEST(LlvmLibcGrpUtilsTest, ParseGroupLine_InvalidNumeric) {
  char *mem_ptrs[4];
  struct group grp;

  char line1[] = "root:x:abc:root";
  EXPECT_FALSE(LIBC_NAMESPACE::grp::parse_group_line(line1, &grp, mem_ptrs));

  char line2[] = "root:x:-1:root";
  EXPECT_FALSE(LIBC_NAMESPACE::grp::parse_group_line(line2, &grp, mem_ptrs));

  char line3[] = "root:x::root";
  EXPECT_FALSE(LIBC_NAMESPACE::grp::parse_group_line(line3, &grp, mem_ptrs));

  char line4[] = "root:x:999999999999999999999999999999:root";
  EXPECT_FALSE(LIBC_NAMESPACE::grp::parse_group_line(line4, &grp, mem_ptrs));
}

TEST(LlvmLibcGrpUtilsTest, ParseGroupLine_MissingFields) {
  char *mem_ptrs[4];
  struct group grp;

  char line1[] = "root:x";
  EXPECT_FALSE(LIBC_NAMESPACE::grp::parse_group_line(line1, &grp, mem_ptrs));

  char line2[] = "root:x:0";
  EXPECT_FALSE(LIBC_NAMESPACE::grp::parse_group_line(line2, &grp, mem_ptrs));
}

TEST(LlvmLibcGrpUtilsTest, ParseGroupLine_EmptyGroupName) {
  char *mem_ptrs[4];
  struct group grp;
  char line[] = ":x:0:root";
  EXPECT_FALSE(LIBC_NAMESPACE::grp::parse_group_line(line, &grp, mem_ptrs));
}

TEST(LlvmLibcGrpUtilsTest, ParseGroupLine_TrailingGarbage) {
  char *mem_ptrs[4];
  struct group grp;
  char line[] = "root:x:0:root:extra";
  EXPECT_FALSE(LIBC_NAMESPACE::grp::parse_group_line(line, &grp, mem_ptrs));
}

TEST(LlvmLibcGrpUtilsTest, ParseGroupLine_NullInput) {
  char *mem_ptrs[4];
  struct group grp;
  LIBC_NAMESPACE::cpp::span<char> empty;
  EXPECT_FALSE(LIBC_NAMESPACE::grp::parse_group_line(empty, &grp, mem_ptrs));

  char line[] = "root:x:0:root";
  EXPECT_FALSE(LIBC_NAMESPACE::grp::parse_group_line(line, nullptr, mem_ptrs));

  LIBC_NAMESPACE::cpp::span<char *> empty_mem;
  EXPECT_FALSE(LIBC_NAMESPACE::grp::parse_group_line(line, &grp, empty_mem));
}

TEST(LlvmLibcGrpUtilsTest, ParseGroupLine_MemberBufferTooSmall) {
  char line[] = "wheel:x:10:root,admin";
  char *mem_ptrs[2]; // Can only hold 1 member + 1 nullptr
  struct group grp;
  EXPECT_FALSE(LIBC_NAMESPACE::grp::parse_group_line(line, &grp, mem_ptrs));
}

TEST(LlvmLibcGrpUtilsTest, ParseLine_FixedBufferErangeWhenNoScratchSpace) {
  // The line fits in buffer, but there is no space left for gr_mem pointers.
  char buffer[] = "wheel:x:10:root,admin";
  struct group grp;
  const auto res = LIBC_NAMESPACE::pwd::parse_line<struct group>(
      LIBC_NAMESPACE::cpp::span<char>(buffer, sizeof(buffer)), {}, &grp);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), ERANGE);
}

TEST(LlvmLibcGrpUtilsTest, ParseLine_EmbeddedNullByteRejected) {
  constexpr const char RAW[] = "wheel:x:10:root\0admin";
  char buffer[128];
  for (size_t i = 0; i < sizeof(RAW); ++i)
    buffer[i] = RAW[i];
  struct group grp;
  const auto res = LIBC_NAMESPACE::pwd::parse_line<struct group>(
      LIBC_NAMESPACE::cpp::span<char>(buffer, sizeof(RAW)),
      LIBC_NAMESPACE::cpp::span<char>(buffer + sizeof(RAW),
                                      sizeof(buffer) - sizeof(RAW)),
      &grp);
  ASSERT_FALSE(res.has_value());
  EXPECT_EQ(res.error(), EINVAL);
}

TEST(LlvmLibcGrpUtilsTest, ParseLine_SuccessWithTailForMemberPointers) {
  constexpr const char *LINE = "wheel:x:10:root,admin,user1,user2";
  constexpr size_t LEN = 33;
  char buffer[128];
  for (size_t i = 0; i <= LEN; ++i)
    buffer[i] = LINE[i];

  struct group grp;
  const auto res = LIBC_NAMESPACE::pwd::parse_line<struct group>(
      LIBC_NAMESPACE::cpp::span<char>(buffer, LEN + 1),
      LIBC_NAMESPACE::cpp::span<char>(buffer + LEN + 1,
                                      sizeof(buffer) - (LEN + 1)),
      &grp);
  ASSERT_TRUE(res.has_value());
  EXPECT_STREQ(grp.gr_name, "wheel");
  EXPECT_STREQ(grp.gr_passwd, "x");
  EXPECT_EQ(grp.gr_gid, static_cast<gid_t>(10));
  ASSERT_NE(grp.gr_mem, nullptr);
  EXPECT_STREQ(grp.gr_mem[0], "root");
  EXPECT_STREQ(grp.gr_mem[1], "admin");
  EXPECT_STREQ(grp.gr_mem[2], "user1");
  EXPECT_STREQ(grp.gr_mem[3], "user2");
  EXPECT_EQ(grp.gr_mem[4], nullptr);
}
