//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for getpwuid_r.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_passwd.h"
#include "hdr/types/uid_t.h"
#include "pwd_test_utils.h"
#include "src/pwd/getpwuid_r.h"
#include "src/pwd/pwd_utils.h"
#include "test/UnitTest/Test.h"

using LlvmLibcGetpwuidRTest = LlvmLibcPwdTest;

TEST_F(LlvmLibcGetpwuidRTest, Success) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n"
                        "daemon:x:2:2:daemon:/sbin:/sbin/nologin\n"
                        "nobody:x:65534:65534:nobody:/nonexistent:/bin/false\n";
  ScopedPasswdFile test_file(
      libc_make_test_file_path("getpwuid_r_success.test"), content);

  struct passwd pwd;
  char buffer[256];
  struct passwd *result = nullptr;

  ASSERT_EQ(
      LIBC_NAMESPACE::getpwuid_r(1, &pwd, buffer, sizeof(buffer), &result), 0);
  ASSERT_EQ(result, &pwd);
  ASSERT_STREQ(pwd.pw_name, "bin");
  ASSERT_EQ(pwd.pw_uid, static_cast<uid_t>(1));
  ASSERT_EQ(pwd.pw_gid, static_cast<gid_t>(1));
  ASSERT_STREQ(pwd.pw_dir, "/bin");
  ASSERT_STREQ(pwd.pw_shell, "/sbin/nologin");

  // Lookup high UID (nobody)
  result = nullptr;
  ASSERT_EQ(
      LIBC_NAMESPACE::getpwuid_r(65534, &pwd, buffer, sizeof(buffer), &result),
      0);
  ASSERT_EQ(result, &pwd);
  ASSERT_STREQ(pwd.pw_name, "nobody");
  ASSERT_EQ(pwd.pw_uid, static_cast<uid_t>(65534));
}

TEST_F(LlvmLibcGetpwuidRTest, RootUidZero) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwuid_r_zero.test"),
                             content);

  struct passwd pwd;
  char buffer[256];
  struct passwd *result = nullptr;

  ASSERT_EQ(
      LIBC_NAMESPACE::getpwuid_r(0, &pwd, buffer, sizeof(buffer), &result), 0);
  ASSERT_EQ(result, &pwd);
  ASSERT_STREQ(pwd.pw_name, "root");
  ASSERT_EQ(pwd.pw_uid, static_cast<uid_t>(0));
}

TEST_F(LlvmLibcGetpwuidRTest, NotFound) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n";
  ScopedPasswdFile test_file(
      libc_make_test_file_path("getpwuid_r_notfound.test"), content);

  struct passwd pwd;
  char buffer[256];
  struct passwd *result = reinterpret_cast<struct passwd *>(0xdeadbeef);

  ASSERT_EQ(
      LIBC_NAMESPACE::getpwuid_r(999, &pwd, buffer, sizeof(buffer), &result),
      0);
  ASSERT_EQ(result, static_cast<struct passwd *>(nullptr));
}

TEST_F(LlvmLibcGetpwuidRTest, BufferTooSmall) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n";
  ScopedPasswdFile test_file(
      libc_make_test_file_path("getpwuid_r_toosmall.test"), content);

  struct passwd pwd;
  char small_buf[8];
  struct passwd *result = reinterpret_cast<struct passwd *>(0xdeadbeef);

  ASSERT_EQ(LIBC_NAMESPACE::getpwuid_r(0, &pwd, small_buf, sizeof(small_buf),
                                       &result),
            ERANGE);
  ASSERT_EQ(result, static_cast<struct passwd *>(nullptr));

  // Single-byte buffer is insufficient and must return ERANGE.
  char tiny_buf[1];
  result = reinterpret_cast<struct passwd *>(0xdeadbeef);
  ASSERT_EQ(
      LIBC_NAMESPACE::getpwuid_r(0, &pwd, tiny_buf, sizeof(tiny_buf), &result),
      ERANGE);
  ASSERT_EQ(result, static_cast<struct passwd *>(nullptr));

  // Zero-byte buffer is insufficient and must return ERANGE.
  result = reinterpret_cast<struct passwd *>(0xdeadbeef);
  ASSERT_EQ(LIBC_NAMESPACE::getpwuid_r(0, &pwd, small_buf, 0, &result), ERANGE);
  ASSERT_EQ(result, static_cast<struct passwd *>(nullptr));

  // Note: passing nullptr for pwd, buffer, or result is undefined behavior per
  // POSIX. The implementation uses LIBC_CRASH_ON_NULLPTR for each pointer, so
  // there are no nullptr tests here in hermetic unit tests.
}

TEST_F(LlvmLibcGetpwuidRTest, BlankLines) {
  const char *content = "\nroot:x:0:0:root:/root:/bin/bash\n\n\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwuid_r_blank.test"),
                             content);

  struct passwd pwd;
  char buffer[256];
  struct passwd *result = nullptr;

  ASSERT_EQ(
      LIBC_NAMESPACE::getpwuid_r(1, &pwd, buffer, sizeof(buffer), &result), 0);
  ASSERT_EQ(result, &pwd);
  ASSERT_STREQ(pwd.pw_name, "bin");
  ASSERT_EQ(pwd.pw_uid, static_cast<uid_t>(1));
}
