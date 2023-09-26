//===-- Unittests for getcwd ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/errno/libc_errno.h"
#include "src/stdlib/getenv.h"
#include "src/unistd/getcwd.h"

#include "test/IntegrationTest/test.h"

#include <stdlib.h> // For malloc and free

using LIBC_NAMESPACE::cpp::string_view;

TEST_MAIN(int argc, char **argv, char **envp) {
  char buffer[1024];
  ASSERT_TRUE(string_view(LIBC_NAMESPACE::getenv("PWD")) ==
              LIBC_NAMESPACE::getcwd(buffer, 1024));

  // nullptr buffer
  char *cwd = LIBC_NAMESPACE::getcwd(nullptr, 0);
  ASSERT_TRUE(string_view(LIBC_NAMESPACE::getenv("PWD")) == cwd);
  free(cwd);

  // Bad size
  cwd = LIBC_NAMESPACE::getcwd(buffer, 0);
  ASSERT_TRUE(cwd == nullptr);
  ASSERT_EQ(libc_errno, EINVAL);
  libc_errno = 0;

  // Insufficient size
  cwd = LIBC_NAMESPACE::getcwd(buffer, 2);
  ASSERT_TRUE(cwd == nullptr);
  int err = libc_errno;
  ASSERT_EQ(err, ERANGE);

  return 0;
}
