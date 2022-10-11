//===-- Unittests for posix_spwan_file_actions_t manipulation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/spawn/file_actions.h"
#include "src/spawn/posix_spawn_file_actions_addclose.h"
#include "src/spawn/posix_spawn_file_actions_adddup2.h"
#include "src/spawn/posix_spawn_file_actions_addopen.h"
#include "src/spawn/posix_spawn_file_actions_destroy.h"
#include "src/spawn/posix_spawn_file_actions_init.h"
#include "utils/UnitTest/Test.h"

#include <errno.h>
#include <spawn.h>
#include <stdint.h>

TEST(LlvmLibcPosixSpawnFileActionsTest, AddActions) {
  posix_spawn_file_actions_t actions;
  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_init(&actions), 0);

  ASSERT_EQ(uintptr_t(actions.__front), uintptr_t(nullptr));
  ASSERT_EQ(uintptr_t(actions.__back), uintptr_t(nullptr));

  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_addclose(&actions, 10), 0);
  ASSERT_NE(uintptr_t(actions.__front), uintptr_t(nullptr));
  ASSERT_NE(uintptr_t(actions.__back), uintptr_t(nullptr));

  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_adddup2(&actions, 11, 12), 0);
  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_addopen(&actions, 13,
                                                          "path/to/file", 0, 0),
            0);

  __llvm_libc::BaseSpawnFileAction *act =
      reinterpret_cast<__llvm_libc::BaseSpawnFileAction *>(actions.__front);
  int action_count = 0;
  while (act != nullptr) {
    ++action_count;
    if (action_count == 1)
      ASSERT_EQ(act->type, __llvm_libc::BaseSpawnFileAction::CLOSE);
    if (action_count == 2)
      ASSERT_EQ(act->type, __llvm_libc::BaseSpawnFileAction::DUP2);
    if (action_count == 3)
      ASSERT_EQ(act->type, __llvm_libc::BaseSpawnFileAction::OPEN);
    act = act->next;
  }
  ASSERT_EQ(action_count, 3);
  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_destroy(&actions), 0);
}

TEST(LlvmLibcPosixSpawnFileActionsTest, InvalidActions) {
  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_addclose(nullptr, 1), EINVAL);
  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_adddup2(nullptr, 1, 2),
            EINVAL);
  ASSERT_EQ(
      __llvm_libc::posix_spawn_file_actions_addopen(nullptr, 1, nullptr, 0, 0),
      EINVAL);
  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_destroy(nullptr), EINVAL);

  posix_spawn_file_actions_t actions;
  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_init(&actions), 0);
  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_addclose(&actions, -1),
            EBADF);
  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_adddup2(&actions, -1, 2),
            EBADF);
  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_adddup2(&actions, 1, -2),
            EBADF);
  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_addopen(&actions, -1, nullptr,
                                                          0, 0),
            EBADF);
  ASSERT_EQ(__llvm_libc::posix_spawn_file_actions_destroy(&actions), 0);
}
