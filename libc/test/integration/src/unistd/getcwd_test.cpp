//===-- Unittests for getcwd ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/stdlib/getenv.h"
#include "src/unistd/getcwd.h"

#include "utils/IntegrationTest/test.h"

#include <errno.h>

using __llvm_libc::cpp::string_view;

TEST_MAIN(int argc, char **argv, char **envp) {
  char buffer[1024];
  ASSERT_TRUE(string_view(__llvm_libc::getenv("PWD")) ==
              __llvm_libc::getcwd(buffer, 1024));

  // nullptr buffer
  char *cwd = __llvm_libc::getcwd(nullptr, 0);
  ASSERT_TRUE(string_view(__llvm_libc::getenv("PWD")) == cwd);
  free(cwd);

  // Bad size
  cwd = __llvm_libc::getcwd(buffer, 0);
  ASSERT_TRUE(cwd == nullptr);
  ASSERT_EQ(errno, EINVAL);
  errno = 0;

  // Insufficient size
  cwd = __llvm_libc::getcwd(buffer, 2);
  ASSERT_TRUE(cwd == nullptr);
  int err = errno;
  ASSERT_EQ(err, ERANGE);

  return 0;
}
