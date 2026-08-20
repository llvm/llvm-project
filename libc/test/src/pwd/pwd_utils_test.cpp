//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for parse_passwd_line.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/struct_passwd.h"
#include "src/pwd/pwd_utils.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcPwdTest, ParsePasswdLine_Success) {
  char line[] = "root:x:0:0:root:/root:/bin/bash";
  auto res = LIBC_NAMESPACE::internal::parse_passwd_line(line);
  ASSERT_TRUE(res.has_value());
  struct passwd pwd = res.value();
  ASSERT_STREQ(pwd.pw_name, "root");
  ASSERT_STREQ(pwd.pw_passwd, "x");
  ASSERT_EQ(pwd.pw_uid, 0u);
  ASSERT_EQ(pwd.pw_gid, 0u);
  ASSERT_STREQ(pwd.pw_gecos, "root");
  ASSERT_STREQ(pwd.pw_dir, "/root");
  ASSERT_STREQ(pwd.pw_shell, "/bin/bash");
}

TEST(LlvmLibcPwdTest, ParsePasswdLine_EmptyFields) {
  char line[] = "root::0:0::/root:";
  auto res = LIBC_NAMESPACE::internal::parse_passwd_line(line);
  ASSERT_TRUE(res.has_value());
  struct passwd pwd = res.value();
  ASSERT_STREQ(pwd.pw_name, "root");
  ASSERT_STREQ(pwd.pw_passwd, "");
  ASSERT_EQ(pwd.pw_uid, 0u);
  ASSERT_EQ(pwd.pw_gid, 0u);
  ASSERT_STREQ(pwd.pw_gecos, "");
  ASSERT_STREQ(pwd.pw_dir, "/root");
  ASSERT_STREQ(pwd.pw_shell, "");
}

TEST(LlvmLibcPwdTest, ParsePasswdLine_InvalidNumeric) {
  char line1[] = "root:x:abc:0:root:/root:/bin/bash";
  auto res1 = LIBC_NAMESPACE::internal::parse_passwd_line(line1);
  ASSERT_FALSE(res1.has_value());
  ASSERT_EQ(res1.error(), EINVAL);

  char line2[] = "root:x:0:def:root:/root:/bin/bash";
  auto res2 = LIBC_NAMESPACE::internal::parse_passwd_line(line2);
  ASSERT_FALSE(res2.has_value());
  ASSERT_EQ(res2.error(), EINVAL);

  char line3[] = "root:x:-1:0:root:/root:/bin/bash";
  auto res3 = LIBC_NAMESPACE::internal::parse_passwd_line(line3);
  ASSERT_FALSE(res3.has_value());
  ASSERT_EQ(res3.error(), EINVAL);

  char line4[] = "root:x:0:-1:root:/root:/bin/bash";
  auto res4 = LIBC_NAMESPACE::internal::parse_passwd_line(line4);
  ASSERT_FALSE(res4.has_value());
  ASSERT_EQ(res4.error(), EINVAL);
}

TEST(LlvmLibcPwdTest, ParsePasswdLine_MissingFields) {
  char line[] = "root:x:0:0:root:/root";
  auto res = LIBC_NAMESPACE::internal::parse_passwd_line(line);
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EINVAL);
}

TEST(LlvmLibcPwdTest, ParsePasswdLine_NullInput) {
  auto res = LIBC_NAMESPACE::internal::parse_passwd_line(nullptr);
  ASSERT_FALSE(res.has_value());
  ASSERT_EQ(res.error(), EINVAL);
}

TEST(LlvmLibcPwdTest, ParsePasswdLine_TrailingGarbage) {
  char line1[] = "root:x:0a:0:root:/root:/bin/bash";
  auto res1 = LIBC_NAMESPACE::internal::parse_passwd_line(line1);
  ASSERT_FALSE(res1.has_value());
  ASSERT_EQ(res1.error(), EINVAL);

  char line2[] = "root:x:0:0b:root:/root:/bin/bash";
  auto res2 = LIBC_NAMESPACE::internal::parse_passwd_line(line2);
  ASSERT_FALSE(res2.has_value());
  ASSERT_EQ(res2.error(), EINVAL);
}

TEST(LlvmLibcPwdTest, ParsePasswdLine_Overflow) {
  char line1[] = "root:x:4294967296:0:root:/root:/bin/bash";
  auto res1 = LIBC_NAMESPACE::internal::parse_passwd_line(line1);
  ASSERT_FALSE(res1.has_value());
  ASSERT_EQ(res1.error(), EINVAL);

  char line2[] = "root:x:0:4294967296:root:/root:/bin/bash";
  auto res2 = LIBC_NAMESPACE::internal::parse_passwd_line(line2);
  ASSERT_FALSE(res2.has_value());
  ASSERT_EQ(res2.error(), EINVAL);
}
