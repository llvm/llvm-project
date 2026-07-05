//===-- Unittests for getcwd ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/stat/stat.h"
#include "src/unistd/getcwd.h"
#include "test/IntegrationTest/test.h"

#include <errno.h>
#include <stdlib.h> // For malloc and free

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          [[maybe_unused]] char **envp) {
  char buffer[1024];
  char *cwd = LIBC_NAMESPACE::getcwd(buffer, sizeof(buffer));
  ASSERT_TRUE(cwd != nullptr);

  struct stat st_dot;
  struct stat st_cwd;

  ASSERT_EQ(LIBC_NAMESPACE::stat(".", &st_dot), 0);
  ASSERT_EQ(LIBC_NAMESPACE::stat(cwd, &st_cwd), 0);

  ASSERT_EQ(st_dot.st_dev, st_cwd.st_dev);
  ASSERT_EQ(st_dot.st_ino, st_cwd.st_ino);

  // nullptr buffer
  cwd = LIBC_NAMESPACE::getcwd(nullptr, 0);
  ASSERT_EQ(LIBC_NAMESPACE::stat(cwd, &st_cwd), 0);
  ASSERT_EQ(st_dot.st_dev, st_cwd.st_dev);
  ASSERT_EQ(st_dot.st_ino, st_cwd.st_ino);
  free(cwd);

  // Bad size
  cwd = LIBC_NAMESPACE::getcwd(buffer, 0);
  ASSERT_TRUE(cwd == nullptr);
  ASSERT_ERRNO_EQ(EINVAL);

  // Insufficient size
  errno = 0;
  cwd = LIBC_NAMESPACE::getcwd(buffer, 2);
  ASSERT_TRUE(cwd == nullptr);
  ASSERT_ERRNO_EQ(ERANGE);

  return 0;
}
