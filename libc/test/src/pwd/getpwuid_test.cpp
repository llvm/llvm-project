//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for getpwuid.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/gid_t.h"
#include "hdr/types/struct_passwd.h"
#include "hdr/types/uid_t.h"
#include "pwd_test_utils.h"
#include "src/__support/libc_errno.h"
#include "src/pwd/getpwuid.h"
#include "src/pwd/pwd_utils.h"
#include "test/UnitTest/Test.h"

using LlvmLibcGetpwuidTest = LlvmLibcPwdTest;

TEST_F(LlvmLibcGetpwuidTest, Success) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n"
                        "daemon:x:2:2:daemon:/sbin:/sbin/nologin\n"
                        "nobody:x:65534:65534:nobody:/nonexistent:/bin/false\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwuid_success.test"),
                             content);

  struct passwd *pwd = LIBC_NAMESPACE::getpwuid(1);
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "bin");
  ASSERT_EQ(pwd->pw_uid, static_cast<uid_t>(1));
  ASSERT_EQ(pwd->pw_gid, static_cast<gid_t>(1));
  ASSERT_STREQ(pwd->pw_dir, "/bin");
  ASSERT_STREQ(pwd->pw_shell, "/sbin/nologin");

  // Lookup high UID (nobody)
  pwd = LIBC_NAMESPACE::getpwuid(65534);
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "nobody");
  ASSERT_EQ(pwd->pw_uid, static_cast<uid_t>(65534));
}

TEST_F(LlvmLibcGetpwuidTest, RootUidZero) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwuid_zero.test"),
                             content);

  struct passwd *pwd = LIBC_NAMESPACE::getpwuid(0);
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "root");
  ASSERT_EQ(pwd->pw_uid, static_cast<uid_t>(0));
}

TEST_F(LlvmLibcGetpwuidTest, NotFound) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwuid_notfound.test"),
                             content);

  // POSIX specifies that errno must not be changed when an entry is not found.
  // Pre-set errno to confirm it remains untouched across the call.
  LIBC_NAMESPACE::libc_errno = ENOENT;
  struct passwd *pwd = LIBC_NAMESPACE::getpwuid(999);
  ASSERT_EQ(pwd, nullptr);
  ASSERT_ERRNO_EQ(ENOENT);

  // When errno is initially zero, it remains zero.
  pwd = LIBC_NAMESPACE::getpwuid(999);
  ASSERT_EQ(pwd, nullptr);
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcGetpwuidTest, BlankLines) {
  const char *content = "\nroot:x:0:0:root:/root:/bin/bash\n\n\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwuid_blank.test"),
                             content);

  struct passwd *pwd = LIBC_NAMESPACE::getpwuid(1);
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "bin");
  ASSERT_EQ(pwd->pw_uid, static_cast<uid_t>(1));
}

TEST_F(LlvmLibcGetpwuidTest, FileOpenFailure) {
  LIBC_NAMESPACE::pwd::TESTONLY_set_passwd_path(
      "/nonexistent_directory/nonexistent_file");

  struct passwd *pwd = LIBC_NAMESPACE::getpwuid(0);
  ASSERT_EQ(pwd, nullptr);
  ASSERT_ERRNO_EQ(ENOENT);
}

TEST_F(LlvmLibcGetpwuidTest, BufferTooSmall) {
  // A line exceeding line_buffer (1024 bytes) triggers ERANGE.
  char content[1100];
  LIBC_NAMESPACE::internal::strlcpy(content,
                                    "longuser:x:1000:1000:", sizeof(content));
  size_t cur = LIBC_NAMESPACE::internal::string_length(content);
  for (; cur < 1050; ++cur)
    content[cur] = 'a';
  LIBC_NAMESPACE::internal::strlcpy(content + cur, ":/home/longuser:/bin/sh\n",
                                    sizeof(content) - cur);

  ScopedPasswdFile test_file(libc_make_test_file_path("getpwuid_toosmall.test"),
                             content);

  struct passwd *pwd = LIBC_NAMESPACE::getpwuid(1000);
  ASSERT_EQ(pwd, nullptr);
  ASSERT_ERRNO_EQ(ERANGE);
}
