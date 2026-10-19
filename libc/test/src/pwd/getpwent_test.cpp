//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for getpwent, setpwent, and endpwent.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/struct_passwd.h"
#include "src/__support/libc_errno.h"
#include "src/pwd/endpwent.h"
#include "src/pwd/getpwent.h"
#include "src/pwd/pwd_utils.h"
#include "src/pwd/setpwent.h"
#include "test/UnitTest/Test.h"
#include "test/src/pwd/pwd_test_utils.h"

TEST_F(LlvmLibcPwdTest, GetPwentTestSuccess) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwent_success.test"),
                             content);

  LIBC_NAMESPACE::setpwent();

  struct passwd *pwd1 = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd1, nullptr);
  ASSERT_STREQ(pwd1->pw_name, "root");
  ASSERT_EQ(pwd1->pw_uid, 0u);

  struct passwd *pwd2 = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd2, nullptr);
  ASSERT_STREQ(pwd2->pw_name, "bin");
  ASSERT_EQ(pwd2->pw_uid, 1u);

  struct passwd *pwd3 = LIBC_NAMESPACE::getpwent();
  ASSERT_EQ(pwd3, nullptr);

  LIBC_NAMESPACE::endpwent();
}

TEST_F(LlvmLibcPwdTest, GetPwentTestFailure) {
  const char *content = "invalid_line_without_enough_fields\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwent_fail.test"),
                             content);

  LIBC_NAMESPACE::setpwent();

  struct passwd *pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_EQ(pwd, nullptr);
  ASSERT_ERRNO_EQ(EINVAL);

  LIBC_NAMESPACE::endpwent();
}

TEST_F(LlvmLibcPwdTest, SetPwentTestHermetic) {
  const char *content = "user1:x:1000:1000:User One:/home/user1:/bin/bash\n"
                        "user2:x:1001:1001:User Two:/home/user2:/bin/bash\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("setpwent_hermetic.test"),
                             content);

  struct passwd *pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "user1");

  pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "user2");

  // Reset iteration
  LIBC_NAMESPACE::setpwent();

  pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "user1");

  LIBC_NAMESPACE::endpwent();
}

TEST_F(LlvmLibcPwdTest, ReopenAfterEndpwent) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("reopen_endpwent.test"),
                             content);

  struct passwd *pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "root");

  LIBC_NAMESPACE::endpwent();

  pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "root");

  LIBC_NAMESPACE::endpwent();
}

TEST_F(LlvmLibcPwdTest, FileOpenFailure) {
  auto missing_path =
      libc_make_test_file_path("nonexistent_dir/getpwent_missing.test");
  LIBC_NAMESPACE::pwd::TESTONLY_set_passwd_path(missing_path);
  LIBC_NAMESPACE::endpwent(); // Force close any existing file

  struct passwd *pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_EQ(pwd, nullptr);
  ASSERT_ERRNO_EQ(ENOENT);
}

TEST_F(LlvmLibcPwdTest, BlankLines) {
  const char *content = "\nroot:x:0:0:root:/root:/bin/bash\n\n\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwent_blank.test"),
                             content);

  LIBC_NAMESPACE::setpwent();
  struct passwd *pwd1 = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd1, nullptr);
  ASSERT_STREQ(pwd1->pw_name, "root");

  struct passwd *pwd2 = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd2, nullptr);
  ASSERT_STREQ(pwd2->pw_name, "bin");

  struct passwd *pwd3 = LIBC_NAMESPACE::getpwent();
  ASSERT_EQ(pwd3, nullptr);

  LIBC_NAMESPACE::endpwent();
}

TEST_F(LlvmLibcPwdTest, LongLineGrowsBuffer) {
  // Record 0 is medium-sized (600 bytes) and record 1 is large (3,000 bytes)
  // so that the second getpwent call must grow the iteration buffer beyond
  // the capacity allocated for the first call.
  constexpr size_t GECOS_LENGTH_0 = 600;
  constexpr size_t GECOS_LENGTH_1 = 3000;
  constexpr size_t RECORD_OVERHEAD = 64;
  constexpr size_t CONTENT_BUFFER_SIZE =
      GECOS_LENGTH_0 + GECOS_LENGTH_1 + 2 * RECORD_OVERHEAD;
  char content[CONTENT_BUFFER_SIZE];

  size_t pos = 0;
  for (size_t record = 0; record < 2; ++record) {
    const char *prefix = record == 0 ? "user0:x:100:100:" : "user1:x:101:101:";
    size_t gecos_len = record == 0 ? GECOS_LENGTH_0 : GECOS_LENGTH_1;
    for (const char *p = prefix; *p != '\0'; ++p)
      content[pos++] = *p;
    for (size_t i = 0; i < gecos_len; ++i)
      content[pos++] = 'g';
    for (const char *p = ":/home/user:/bin/sh\n"; *p != '\0'; ++p)
      content[pos++] = *p;
  }
  content[pos] = '\0';

  ScopedPasswdFile test_file(libc_make_test_file_path("getpwent_longline.test"),
                             content);

  LIBC_NAMESPACE::setpwent();

  struct passwd *pwd1 = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd1, nullptr);
  ASSERT_STREQ(pwd1->pw_name, "user0");
  ASSERT_EQ(LIBC_NAMESPACE::internal::string_length(pwd1->pw_gecos),
            GECOS_LENGTH_0);

  struct passwd *pwd2 = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd2, nullptr);
  ASSERT_STREQ(pwd2->pw_name, "user1");
  ASSERT_EQ(LIBC_NAMESPACE::internal::string_length(pwd2->pw_gecos),
            GECOS_LENGTH_1);
  ASSERT_STREQ(pwd2->pw_shell, "/bin/sh");

  ASSERT_EQ(LIBC_NAMESPACE::getpwent(), nullptr);

  LIBC_NAMESPACE::endpwent();
}

TEST_F(LlvmLibcPwdTest, EndPwentClosesStreamAndIterationRestarts) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwent_reopen.test"),
                             content);

  LIBC_NAMESPACE::setpwent();
  struct passwd *pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "root");

  // endpwent closes the file stream without freeing the static buffer, so the
  // last returned pointer remains valid and the next iteration reopens from
  // the top.
  LIBC_NAMESPACE::endpwent();
  ASSERT_STREQ(pwd->pw_name, "root");

  pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_NE(pwd, nullptr);
  ASSERT_STREQ(pwd->pw_name, "root");

  LIBC_NAMESPACE::endpwent();
}
