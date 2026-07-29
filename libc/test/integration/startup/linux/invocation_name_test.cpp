//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration test for program_invocation_name.
///
//===----------------------------------------------------------------------===//

#include "src/errno/program_invocation_name.h"
#include "src/errno/program_invocation_short_name.h"
#include "src/unistd/execve.h"
#include "test/IntegrationTest/test.h"

static bool my_streq(const char *lhs, const char *rhs) {
  const char *l, *r;
  for (l = lhs, r = rhs; *l != '\0' && *r != '\0'; ++l, ++r)
    if (*l != *r)
      return false;

  return *l == '\0' && *r == '\0';
}

TEST_MAIN(int argc, char **argv, char **envp) {
  ASSERT_TRUE(LIBC_NAMESPACE::program_invocation_name != nullptr);
  ASSERT_TRUE(LIBC_NAMESPACE::program_invocation_short_name != nullptr);
  ASSERT_TRUE(LIBC_NAMESPACE::program_invocation_name == argv[0]);

  if (argc == 3 && my_streq(argv[1], "reexec1")) {
    // Step 1: Executed with an absolute path containing slashes.
    ASSERT_TRUE(my_streq(LIBC_NAMESPACE::program_invocation_name,
                         "/known/path/to/invocation_name"));
    ASSERT_TRUE(my_streq(LIBC_NAMESPACE::program_invocation_short_name,
                         "invocation_name"));

    const char *self_path = argv[2];
    char *const child_argv[] = {
        const_cast<char *>("./relative/path/to/invocation_name"),
        const_cast<char *>("reexec2"),
        const_cast<char *>(self_path),
        nullptr,
    };
    LIBC_NAMESPACE::execve(self_path, child_argv, envp);
    ASSERT_TRUE(false);
  }

  if (argc == 3 && my_streq(argv[1], "reexec2")) {
    // Step 2: Executed with a relative path containing slashes.
    ASSERT_TRUE(my_streq(LIBC_NAMESPACE::program_invocation_name,
                         "./relative/path/to/invocation_name"));
    ASSERT_TRUE(my_streq(LIBC_NAMESPACE::program_invocation_short_name,
                         "invocation_name"));

    const char *self_path = argv[2];
    char *const child_argv[] = {
        const_cast<char *>(""),
        const_cast<char *>("reexec3"),
        const_cast<char *>(self_path),
        nullptr,
    };
    LIBC_NAMESPACE::execve(self_path, child_argv, envp);
    ASSERT_TRUE(false);
  }

  if (argc == 3 && my_streq(argv[1], "reexec3")) {
    // Step 3: Executed with an empty string.
    ASSERT_TRUE(LIBC_NAMESPACE::program_invocation_name ==
                LIBC_NAMESPACE::program_invocation_short_name);
    ASSERT_TRUE(my_streq(LIBC_NAMESPACE::program_invocation_name, ""));
    ASSERT_TRUE(my_streq(LIBC_NAMESPACE::program_invocation_short_name, ""));

    const char *self_path = argv[2];
    char *const child_argv[] = {
        const_cast<char *>("/known/path/to/dir/"),
        const_cast<char *>("reexec4"),
        const_cast<char *>(self_path),
        nullptr,
    };
    LIBC_NAMESPACE::execve(self_path, child_argv, envp);
    ASSERT_TRUE(false);
  }

  if (argc == 3 && my_streq(argv[1], "reexec4")) {
    // Step 4: Executed with a path ending in a slash.
    ASSERT_TRUE(my_streq(LIBC_NAMESPACE::program_invocation_name,
                         "/known/path/to/dir/"));
    ASSERT_TRUE(my_streq(LIBC_NAMESPACE::program_invocation_short_name, ""));

    const char *self_path = argv[2];
    char *const child_argv[] = {
        const_cast<char *>("invocation_name_no_slash"),
        const_cast<char *>("reexec5"),
        nullptr,
    };
    LIBC_NAMESPACE::execve(self_path, child_argv, envp);
    ASSERT_TRUE(false);
  }

  if (argc == 2 && my_streq(argv[1], "reexec5")) {
    // Step 5: Executed with a path without slashes.
    ASSERT_TRUE(LIBC_NAMESPACE::program_invocation_name ==
                LIBC_NAMESPACE::program_invocation_short_name);
    ASSERT_TRUE(my_streq(LIBC_NAMESPACE::program_invocation_name,
                         "invocation_name_no_slash"));
    ASSERT_TRUE(my_streq(LIBC_NAMESPACE::program_invocation_short_name,
                         "invocation_name_no_slash"));
    return 0;
  }

  // Step 0: Initial run. Re-exec self with known argv[0] values.
  const char *self_path = argv[0];
  char *const child_argv[] = {
      const_cast<char *>("/known/path/to/invocation_name"),
      const_cast<char *>("reexec1"),
      const_cast<char *>(self_path),
      nullptr,
  };
  LIBC_NAMESPACE::execve(self_path, child_argv, envp);
  ASSERT_TRUE(false);
  return 1;
}
