//===-- Unittests for getenv ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/getenv.h"

#include "utils/IntegrationTest/test.h"

static bool my_streq(const char *lhs, const char *rhs) {
  if (lhs == rhs)
    return true;
  if (((lhs == static_cast<char *>(nullptr)) &&
       (rhs != static_cast<char *>(nullptr))) ||
      ((lhs != static_cast<char *>(nullptr)) &&
       (rhs == static_cast<char *>(nullptr)))) {
    return false;
  }
  const char *l, *r;
  for (l = lhs, r = rhs; *l != '\0' && *r != '\0'; ++l, ++r)
    if (*l != *r)
      return false;

  return *l == '\0' && *r == '\0';
}

TEST_MAIN(int argc, char **argv, char **envp) {
  ASSERT_TRUE(my_streq(__llvm_libc::getenv(""), static_cast<char *>(nullptr)));
  ASSERT_TRUE(my_streq(__llvm_libc::getenv("="), static_cast<char *>(nullptr)));
  ASSERT_TRUE(my_streq(__llvm_libc::getenv("MISSING ENV VARIABLE"),
                       static_cast<char *>(nullptr)));
  ASSERT_FALSE(
      my_streq(__llvm_libc::getenv("PATH"), static_cast<char *>(nullptr)));
  ASSERT_TRUE(my_streq(__llvm_libc::getenv("FRANCE"), "Paris"));
  ASSERT_FALSE(my_streq(__llvm_libc::getenv("FRANCE"), "Berlin"));
  ASSERT_TRUE(my_streq(__llvm_libc::getenv("GERMANY"), "Berlin"));
  ASSERT_TRUE(
      my_streq(__llvm_libc::getenv("FRANC"), static_cast<char *>(nullptr)));
  ASSERT_TRUE(
      my_streq(__llvm_libc::getenv("FRANCE1"), static_cast<char *>(nullptr)));

  return 0;
}
