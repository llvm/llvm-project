//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for getpwnam.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/struct_passwd.h"
#include "hdr/types/uid_t.h"
#include "pwd_test_utils.h"
#include "src/__support/libc_errno.h"
#include "src/pwd/getpwnam.h"
#include "src/pwd/pwd_utils.h"
#include "test/UnitTest/Test.h"

using LlvmLibcGetpwnamTest = LlvmLibcPwdTest;

TEST_F(LlvmLibcGetpwnamTest, Success) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n"
                        "daemon:x:2:2:daemon:/sbin:/sbin/nologin\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwnam_success.test"),
                             content);

  struct passwd *pwd = LIBC_NAMESPACE::getpwnam("bin");
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "bin");
  ASSERT_STREQ(pwd->pw_passwd, "x");
  ASSERT_EQ(pwd->pw_uid, static_cast<uid_t>(1));
  ASSERT_EQ(pwd->pw_gid, static_cast<gid_t>(1));
  ASSERT_STREQ(pwd->pw_gecos, "bin");
  ASSERT_STREQ(pwd->pw_dir, "/bin");
  ASSERT_STREQ(pwd->pw_shell, "/sbin/nologin");
}

TEST_F(LlvmLibcGetpwnamTest, FirstAndLastEntries) {
  const char *content = "first:x:100:100:first:/home/first:/bin/sh\n"
                        "middle:x:101:101:middle:/home/middle:/bin/sh\n"
                        "last:x:102:102:last:/home/last:/bin/sh\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwnam_boundary.test"),
                             content);

  struct passwd *pwd = LIBC_NAMESPACE::getpwnam("first");
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "first");
  ASSERT_EQ(pwd->pw_uid, static_cast<uid_t>(100));

  pwd = LIBC_NAMESPACE::getpwnam("last");
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "last");
  ASSERT_EQ(pwd->pw_uid, static_cast<uid_t>(102));
}

TEST_F(LlvmLibcGetpwnamTest, NotFound) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwnam_notfound.test"),
                             content);

  // POSIX specifies that errno must not be changed when an entry is not found.
  // Pre-set errno to confirm it remains untouched across the call.
  LIBC_NAMESPACE::libc_errno = ENOENT;
  struct passwd *pwd = LIBC_NAMESPACE::getpwnam("nonexistent");
  ASSERT_EQ(pwd, nullptr);
  ASSERT_ERRNO_EQ(ENOENT);

  // When errno is initially zero, it remains zero.
  pwd = LIBC_NAMESPACE::getpwnam("nonexistent");
  ASSERT_EQ(pwd, nullptr);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcGetpwnamTest, BlankLines) {
  const char *content = "\nroot:x:0:0:root:/root:/bin/bash\n\n\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwnam_blank.test"),
                             content);

  struct passwd *pwd = LIBC_NAMESPACE::getpwnam("bin");
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "bin");
  ASSERT_EQ(pwd->pw_uid, static_cast<uid_t>(1));
}

TEST_F(LlvmLibcGetpwnamTest, FileOpenFailure) {
  LIBC_NAMESPACE::pwd::TESTONLY_set_passwd_path(
      "/nonexistent_directory/nonexistent_file");

  struct passwd *pwd = LIBC_NAMESPACE::getpwnam("root");
  ASSERT_EQ(pwd, nullptr);
  ASSERT_ERRNO_EQ(ENOENT);
}

TEST_F(LlvmLibcGetpwnamTest, BufferTooSmall) {
  // A line exceeding line_buffer (1024 bytes) triggers ERANGE.
  char content[1100];
  LIBC_NAMESPACE::internal::strlcpy(content,
                                    "longuser:x:1000:1000:", sizeof(content));
  size_t cur = LIBC_NAMESPACE::internal::string_length(content);
  for (; cur < 1050; ++cur)
    content[cur] = 'a';
  LIBC_NAMESPACE::internal::strlcpy(content + cur, ":/home/longuser:/bin/sh\n",
                                    sizeof(content) - cur);

  ScopedPasswdFile test_file(libc_make_test_file_path("getpwnam_toosmall.test"),
                             content);

  struct passwd *pwd = LIBC_NAMESPACE::getpwnam("longuser");
  ASSERT_EQ(pwd, nullptr);
  ASSERT_ERRNO_EQ(ERANGE);
}
