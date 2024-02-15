//===-- Unittests for getcwd ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/IntegrationTest/test.h"

#include <errno.h>  // errno
#include <stdlib.h> // getenv
#include <unistd.h> // getcwd

TEST_MAIN(int argc, char **argv, char **envp) {
  char buffer[1024];
  ASSERT_STREQ(getenv("PWD"), getcwd(buffer, 1024));

  // nullptr buffer
  char *cwd = getcwd(nullptr, 0);
  ASSERT_STREQ(getenv("PWD"), cwd);
  free(cwd);

  // Bad size
  cwd = getcwd(buffer, 0);
  ASSERT_TRUE(cwd == nullptr);
  ASSERT_ERRNO_EQ(EINVAL);
  LIBC_NAMESPACE::libc_errno = 0;

  // Insufficient size
  cwd = getcwd(buffer, 2);
  ASSERT_TRUE(cwd == nullptr);
  int err = LIBC_NAMESPACE::libc_errno;
  ASSERT_EQ(err, ERANGE);

  return 0;
}
