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
#include "pwd_test_utils.h"
#include "src/__support/libc_errno.h"
#include "src/pwd/endpwent.h"
#include "src/pwd/getpwent.h"
#include "src/pwd/pwd_utils.h"
#include "src/pwd/setpwent.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST_F(LlvmLibcPwdTest, GetPwentTestSuccess) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwent_success.test"),
                             content);

  LIBC_NAMESPACE::setpwent();

  struct passwd *pwd1 = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pwd1 != nullptr);
  ASSERT_STREQ(pwd1->pw_name, "root");
  ASSERT_EQ(pwd1->pw_uid, 0u);

  struct passwd *pwd2 = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pwd2 != nullptr);
  ASSERT_STREQ(pwd2->pw_name, "bin");
  ASSERT_EQ(pwd2->pw_uid, 1u);

  struct passwd *pwd3 = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pwd3 == nullptr);

  LIBC_NAMESPACE::endpwent();
}

TEST_F(LlvmLibcPwdTest, GetPwentTestFailure) {
  const char *content = "invalid_line_without_enough_fields\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwent_fail.test"),
                             content);

  LIBC_NAMESPACE::setpwent();

  struct passwd *pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pwd == nullptr);
  ASSERT_ERRNO_EQ(EINVAL);

  LIBC_NAMESPACE::endpwent();
}

TEST_F(LlvmLibcPwdTest, SetPwentTestHermetic) {
  const char *content = "user1:x:1000:1000:User One:/home/user1:/bin/bash\n"
                        "user2:x:1001:1001:User Two:/home/user2:/bin/bash\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("setpwent_hermetic.test"),
                             content);

  struct passwd *pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pwd != nullptr);
  ASSERT_STREQ(pwd->pw_name, "user1");

  pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pwd != nullptr);
  ASSERT_STREQ(pwd->pw_name, "user2");

  // Reset iteration
  LIBC_NAMESPACE::setpwent();

  pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pwd != nullptr);
  ASSERT_STREQ(pwd->pw_name, "user1");

  LIBC_NAMESPACE::endpwent();
}

TEST_F(LlvmLibcPwdTest, ReopenAfterEndpwent) {
  const char *content = "root:x:0:0:root:/root:/bin/bash\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("reopen_endpwent.test"),
                             content);

  struct passwd *pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pwd != nullptr);
  ASSERT_STREQ(pwd->pw_name, "root");

  LIBC_NAMESPACE::endpwent();

  pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pwd != nullptr);
  ASSERT_STREQ(pwd->pw_name, "root");

  LIBC_NAMESPACE::endpwent();
}

TEST_F(LlvmLibcPwdTest, FileOpenFailure) {
  LIBC_NAMESPACE::pwd::TESTONLY_set_passwd_path(
      "/nonexistent_directory/nonexistent_file");
  LIBC_NAMESPACE::endpwent(); // Force close any existing file

  struct passwd *pwd = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pwd == nullptr);
  ASSERT_ERRNO_EQ(ENOENT);
}

TEST_F(LlvmLibcPwdTest, BlankLines) {
  const char *content = "\nroot:x:0:0:root:/root:/bin/bash\n\n\n"
                        "bin:x:1:1:bin:/bin:/sbin/nologin\n\n";
  ScopedPasswdFile test_file(libc_make_test_file_path("getpwent_blank.test"),
                             content);

  LIBC_NAMESPACE::setpwent();
  struct passwd *pwd1 = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pwd1 != nullptr);
  ASSERT_STREQ(pwd1->pw_name, "root");

  struct passwd *pwd2 = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pwd2 != nullptr);
  ASSERT_STREQ(pwd2->pw_name, "bin");

  struct passwd *pwd3 = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pwd3 == nullptr);

  LIBC_NAMESPACE::endpwent();
}
