//===-- Unittests for posix_spawn -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test_binary_properties.h"

#include "src/spawn/posix_spawn.h"
#include "src/spawn/posix_spawn_file_actions_addopen.h"
#include "src/spawn/posix_spawn_file_actions_destroy.h"
#include "src/spawn/posix_spawn_file_actions_init.h"
#include "src/sys/wait/waitpid.h"
#include "utils/IntegrationTest/test.h"

#include <fcntl.h>
#include <spawn.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/wait.h>

char arg0[] = "libc_posix_spawn_test_binary";
char *argv[] = {
    arg0,
    nullptr,
};

void spawn_and_wait_for_normal_exit(char **envp) {
  pid_t cpid;
  posix_spawn_file_actions_t file_actions;
  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_init(&file_actions), 0);
  __llvm_libc::posix_spawn_file_actions_addopen(
      &file_actions, CHILD_FD, "testdata/posix_spawn.test", O_RDONLY, 0);
  ASSERT_EQ(
      __llvm_libc::posix_spawn(&cpid, arg0, &file_actions, nullptr, argv, envp),
      0);
  ASSERT_TRUE(cpid > 0);
  int status;
  ASSERT_EQ(__llvm_libc::waitpid(cpid, &status, 0), cpid);
  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_destroy(&file_actions), 0);
  ASSERT_TRUE(WIFEXITED(status));
  int exit_status = WEXITSTATUS(status);
  ASSERT_EQ(exit_status, 0);
}

TEST_MAIN(int argc, char **argv, char **envp) {
  spawn_and_wait_for_normal_exit(envp);
  return 0;
}
