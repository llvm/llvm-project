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

#include "src/__support/CPP/string_view.h"
#include "src/errno/program_invocation_name.h"
#include "src/errno/program_invocation_short_name.h"
#include "src/unistd/execve.h"
#include "test/IntegrationTest/test.h"

TEST_MAIN(int argc, char **argv, char **envp) {
  ASSERT_TRUE(argc >= 1);
  ASSERT_TRUE(LIBC_NAMESPACE::program_invocation_name != nullptr);
  ASSERT_TRUE(LIBC_NAMESPACE::program_invocation_short_name != nullptr);
  ASSERT_TRUE(LIBC_NAMESPACE::program_invocation_name == argv[0]);

  LIBC_NAMESPACE::cpp::string_view arg1 = argc > 1 ? argv[1] : "";

  if (arg1 == "reexec1") {
    // Step 1: Executed with an absolute path containing slashes.
    ASSERT_STREQ(LIBC_NAMESPACE::program_invocation_name,
                 "/known/path/to/invocation_name");
    ASSERT_STREQ(LIBC_NAMESPACE::program_invocation_short_name,
                 "invocation_name");

    ASSERT_EQ(argc, 3);
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

  if (arg1 == "reexec2") {
    // Step 2: Executed with a relative path containing slashes.
    ASSERT_STREQ(LIBC_NAMESPACE::program_invocation_name,
                 "./relative/path/to/invocation_name");
    ASSERT_STREQ(LIBC_NAMESPACE::program_invocation_short_name,
                 "invocation_name");

    ASSERT_EQ(argc, 3);
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

  if (arg1 == "reexec3") {
    // Step 3: Executed with an empty string.
    ASSERT_TRUE(LIBC_NAMESPACE::program_invocation_name ==
                LIBC_NAMESPACE::program_invocation_short_name);
    ASSERT_STREQ(LIBC_NAMESPACE::program_invocation_name, "");
    ASSERT_STREQ(LIBC_NAMESPACE::program_invocation_short_name, "");

    ASSERT_EQ(argc, 3);
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

  if (arg1 == "reexec4") {
    // Step 4: Executed with a path ending in a slash.
    ASSERT_STREQ(LIBC_NAMESPACE::program_invocation_name,
                 "/known/path/to/dir/");
    ASSERT_STREQ(LIBC_NAMESPACE::program_invocation_short_name, "");

    ASSERT_EQ(argc, 3);
    const char *self_path = argv[2];
    char *const child_argv[] = {
        const_cast<char *>("invocation_name_no_slash"),
        const_cast<char *>("reexec5"),
        const_cast<char *>(self_path),
        nullptr,
    };
    LIBC_NAMESPACE::execve(self_path, child_argv, envp);
    ASSERT_TRUE(false);
  }

  if (arg1 == "reexec5") {
    // Step 5: Executed with a path without slashes.
    ASSERT_TRUE(LIBC_NAMESPACE::program_invocation_name ==
                LIBC_NAMESPACE::program_invocation_short_name);
    ASSERT_STREQ(LIBC_NAMESPACE::program_invocation_name,
                 "invocation_name_no_slash");
    ASSERT_STREQ(LIBC_NAMESPACE::program_invocation_short_name,
                 "invocation_name_no_slash");

    ASSERT_EQ(argc, 3);
    const char *self_path = argv[2];
    char *const child_argv[] = {
        nullptr,
    };
    LIBC_NAMESPACE::execve(self_path, child_argv, envp);
    ASSERT_TRUE(false);
  }

  if (argc == 1 && LIBC_NAMESPACE::cpp::string_view(argv[0]) == "") {
    // Step 6: Executed with an empty (zero-length) argv. The kernel adds an
    // empty string for argv[0].
    ASSERT_STREQ(LIBC_NAMESPACE::program_invocation_name, "");
    ASSERT_STREQ(LIBC_NAMESPACE::program_invocation_short_name, "");
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
