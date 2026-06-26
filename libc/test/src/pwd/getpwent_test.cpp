//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getpwent.
///
//===----------------------------------------------------------------------===//

#include "src/pwd/endpwent.h"
#include "src/pwd/getpwent.h"
#include "src/pwd/setpwent.h"
#include "src/string/strcmp.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcPwdTest, GetPwentTest) {
  // We assume /etc/passwd exists and contains at least "root" and "daemon".
  // We also assume it does NOT contain "baduser".

  bool found_root = false;
  bool found_daemon = false;
  bool found_baduser = false;

  LIBC_NAMESPACE::setpwent();

  struct passwd *pw;
  while ((pw = LIBC_NAMESPACE::getpwent()) != nullptr) {
    if (LIBC_NAMESPACE::strcmp(pw->pw_name, "root") == 0) {
      found_root = true;
      ASSERT_EQ(pw->pw_uid, static_cast<uid_t>(0));
    }
    if (LIBC_NAMESPACE::strcmp(pw->pw_name, "daemon") == 0) {
      found_daemon = true;
    }
    if (LIBC_NAMESPACE::strcmp(pw->pw_name, "baduser") == 0) {
      found_baduser = true;
    }
  }

  ASSERT_TRUE(found_root);
  ASSERT_TRUE(found_daemon);
  ASSERT_FALSE(found_baduser);

  LIBC_NAMESPACE::endpwent();
}

TEST(LlvmLibcPwdTest, SetPwentTest) {
  // Read first entry
  LIBC_NAMESPACE::setpwent();
  struct passwd *pw1 = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pw1 != nullptr);

  // Rewind
  LIBC_NAMESPACE::setpwent();
  struct passwd *pw2 = LIBC_NAMESPACE::getpwent();
  ASSERT_TRUE(pw2 != nullptr);

  // Should be the same entry
  ASSERT_STREQ(pw1->pw_name, pw2->pw_name);

  LIBC_NAMESPACE::endpwent();
}
