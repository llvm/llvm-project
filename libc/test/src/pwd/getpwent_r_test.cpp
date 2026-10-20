//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for getpwent_r.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_passwd.h"
#include "hdr/types/uid_t.h"
#include "pwd_test_utils.h"
#include "src/pwd/endpwent.h"
#include "src/pwd/getpwent_r.h"
#include "src/pwd/pwd_utils.h"
#include "src/pwd/setpwent.h"
#include "test/UnitTest/Test.h"

using LlvmLibcGetpwentRTest = LlvmLibcPwdTest;

TEST_F(LlvmLibcGetpwentRTest, SuccessAndEofReturnsEnoent) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n";
  ScopedPasswdFile test_file(
      libc_make_test_file_path("getpwent_r_success.test"), content);

  struct passwd pwd;
  char buffer[256];
  struct passwd *result = nullptr;

  ASSERT_EQ(LIBC_NAMESPACE::getpwent_r(&pwd, buffer, sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &pwd);
  EXPECT_STREQ(pwd.pw_name, "root");
  EXPECT_STREQ(pwd.pw_passwd, "x");
  EXPECT_EQ(pwd.pw_uid, static_cast<uid_t>(0));
  EXPECT_EQ(pwd.pw_gid, static_cast<gid_t>(0));
  EXPECT_STREQ(pwd.pw_gecos, "root");
  EXPECT_STREQ(pwd.pw_dir, "/root");
  EXPECT_STREQ(pwd.pw_shell, "/bin/bash");

  result = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::getpwent_r(&pwd, buffer, sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &pwd);
  EXPECT_STREQ(pwd.pw_name, "bin");
  EXPECT_EQ(pwd.pw_uid, static_cast<uid_t>(1));

  result = reinterpret_cast<struct passwd *>(0xdeadbeef);
  ASSERT_EQ(LIBC_NAMESPACE::getpwent_r(&pwd, buffer, sizeof(buffer), &result),
            ENOENT);
  EXPECT_EQ(result, static_cast<struct passwd *>(nullptr));
}

TEST_F(LlvmLibcGetpwentRTest, SetpwentRewindsAndEndpwentReopens) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwent_r_rewind.test"),
                             content);

  struct passwd pwd;
  char buffer[256];
  struct passwd *result = nullptr;

  ASSERT_EQ(LIBC_NAMESPACE::getpwent_r(&pwd, buffer, sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &pwd);
  EXPECT_STREQ(pwd.pw_name, "root");

  LIBC_NAMESPACE::setpwent();
  result = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::getpwent_r(&pwd, buffer, sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &pwd);
  EXPECT_STREQ(pwd.pw_name, "root");

  LIBC_NAMESPACE::endpwent();
  result = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::getpwent_r(&pwd, buffer, sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &pwd);
  EXPECT_STREQ(pwd.pw_name, "root");
}

TEST_F(LlvmLibcGetpwentRTest, BufferTooSmallAndRetryReadsSameEntry) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwent_r_erange.test"),
                             content);

  struct passwd pwd;
  char small_buf[8];
  struct passwd *result = reinterpret_cast<struct passwd *>(0xdeadbeef);

  ASSERT_EQ(
      LIBC_NAMESPACE::getpwent_r(&pwd, small_buf, sizeof(small_buf), &result),
      ERANGE);
  EXPECT_EQ(result, static_cast<struct passwd *>(nullptr));

  result = reinterpret_cast<struct passwd *>(0xdeadbeef);
  ASSERT_EQ(LIBC_NAMESPACE::getpwent_r(&pwd, small_buf, 0, &result), ERANGE);
  EXPECT_EQ(result, static_cast<struct passwd *>(nullptr));

  // Retrying with a sufficiently large buffer must read the first record
  // ("root") rather than skipping past it.
  char large_buf[256];
  ASSERT_EQ(
      LIBC_NAMESPACE::getpwent_r(&pwd, large_buf, sizeof(large_buf), &result),
      0);
  ASSERT_EQ(result, &pwd);
  EXPECT_STREQ(pwd.pw_name, "root");
}

TEST_F(LlvmLibcGetpwentRTest, BlankLinesSkipped) {
  const char *content = "\nroot:x:0:0:root:/root:/bin/bash\n\n\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwent_r_blank.test"),
                             content);

  struct passwd pwd;
  char buffer[256];
  struct passwd *result = nullptr;

  ASSERT_EQ(LIBC_NAMESPACE::getpwent_r(&pwd, buffer, sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &pwd);
  EXPECT_STREQ(pwd.pw_name, "root");

  ASSERT_EQ(LIBC_NAMESPACE::getpwent_r(&pwd, buffer, sizeof(buffer), &result),
            0);
  ASSERT_EQ(result, &pwd);
  EXPECT_STREQ(pwd.pw_name, "bin");

  ASSERT_EQ(LIBC_NAMESPACE::getpwent_r(&pwd, buffer, sizeof(buffer), &result),
            ENOENT);
  EXPECT_EQ(result, static_cast<struct passwd *>(nullptr));
}

TEST_F(LlvmLibcGetpwentRTest, FileOpenFailure) {
  const auto missing_path =
      libc_make_test_file_path("nonexistent_dir/getpwent_r_missing.test");
  LIBC_NAMESPACE::pwd::TESTONLY_set_passwd_path(missing_path);

  struct passwd pwd;
  char buffer[256];
  struct passwd *result = reinterpret_cast<struct passwd *>(0xdeadbeef);

  ASSERT_EQ(LIBC_NAMESPACE::getpwent_r(&pwd, buffer, sizeof(buffer), &result),
            ENOENT);
  EXPECT_EQ(result, static_cast<struct passwd *>(nullptr));
}

#if defined(LIBC_ADD_NULL_CHECKS) && !defined(LIBC_TESTS_CAN_USE_MPFR)
TEST_F(LlvmLibcGetpwentRTest, NullPointerCrash) {
  struct passwd pwd;
  char buffer[256];
  struct passwd *result = nullptr;
  ASSERT_DEATH(
      [=]() {
        LIBC_NAMESPACE::getpwent_r(nullptr, const_cast<char *>(buffer),
                                   sizeof(buffer),
                                   const_cast<struct passwd **>(&result));
      },
      WITH_SIGNAL(-1));
  ASSERT_DEATH(
      [=]() {
        LIBC_NAMESPACE::getpwent_r(const_cast<struct passwd *>(&pwd), nullptr,
                                   sizeof(buffer),
                                   const_cast<struct passwd **>(&result));
      },
      WITH_SIGNAL(-1));
  ASSERT_DEATH(
      [=]() {
        LIBC_NAMESPACE::getpwent_r(const_cast<struct passwd *>(&pwd),
                                   const_cast<char *>(buffer), sizeof(buffer),
                                   nullptr);
      },
      WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS && !LIBC_TESTS_CAN_USE_MPFR
