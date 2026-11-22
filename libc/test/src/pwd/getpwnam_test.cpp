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
#include "src/__support/libc_errno.h"
#include "src/pwd/endpwent.h"
#include "src/pwd/getpwent.h"
#include "src/pwd/getpwnam.h"
#include "src/pwd/pwd_utils.h"
#include "test/UnitTest/Test.h"
#include "test/src/pwd/pwd_test_utils.h"

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
  auto missing_path =
      libc_make_test_file_path("nonexistent_dir/getpwnam_missing.test");
  LIBC_NAMESPACE::pwd::TESTONLY_set_passwd_path(missing_path);

  struct passwd *pwd = LIBC_NAMESPACE::getpwnam("root");
  ASSERT_EQ(pwd, nullptr);
  ASSERT_ERRNO_EQ(ENOENT);
}

TEST_F(LlvmLibcGetpwnamTest, LongLineGrowsBuffer) {
  // A multi-kilobyte record requires dynamic growth of the lookup buffer to
  // return the entry in full.
  constexpr size_t GECOS_LENGTH = 3000;
  constexpr size_t CONTENT_BUFFER_SIZE = GECOS_LENGTH + 128;
  char content[CONTENT_BUFFER_SIZE];
  LIBC_NAMESPACE::internal::strlcpy(content,
                                    "longuser:x:1000:1000:", sizeof(content));
  size_t prefix_len = LIBC_NAMESPACE::internal::string_length(content);
  size_t cur = prefix_len;
  for (size_t i = 0; i < GECOS_LENGTH; ++i)
    content[cur++] = 'a';
  LIBC_NAMESPACE::internal::strlcpy(content + cur, ":/home/longuser:/bin/sh\n",
                                    sizeof(content) - cur);

  ScopedPasswdFile test_file(libc_make_test_file_path("getpwnam_longline.test"),
                             content);

  struct passwd *pwd = LIBC_NAMESPACE::getpwnam("longuser");
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "longuser");
  ASSERT_EQ(pwd->pw_uid, static_cast<uid_t>(1000));
  ASSERT_STREQ(pwd->pw_dir, "/home/longuser");
  ASSERT_STREQ(pwd->pw_shell, "/bin/sh");
  ASSERT_EQ(LIBC_NAMESPACE::internal::string_length(pwd->pw_gecos),
            GECOS_LENGTH);
}

TEST_F(LlvmLibcGetpwnamTest, DoesNotDisturbIteration) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n"
                        "daemon:x:2:2:daemon:/sbin:/sbin/nologin\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwnam_iter.test"),
                             content);

  struct passwd *pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "root");

  struct passwd *found = LIBC_NAMESPACE::getpwnam("daemon");
  ASSERT_NE(found, nullptr);
  ASSERT_STREQ(found->pw_name, "daemon");

  // Closing the iteration stream via endpwent does not invalidate the pointer
  // returned by getpwnam.
  LIBC_NAMESPACE::endpwent();
  ASSERT_STREQ(found->pw_name, "daemon");

  // Reopening iteration starts from the beginning.
  pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "root");

  // A mid-iteration getpwnam lookup opens its own scoped stream so the
  // getpwent stream position is undisturbed.
  found = LIBC_NAMESPACE::getpwnam("daemon");
  ASSERT_NE(found, nullptr);
  ASSERT_STREQ(found->pw_name, "daemon");

  pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "bin");

  LIBC_NAMESPACE::endpwent();
}
