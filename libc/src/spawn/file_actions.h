//===-- Spawn file actions  -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SPAWN_FILE_ACTIONS_H
#define LLVM_LIBC_SRC_SPAWN_FILE_ACTIONS_H

#include <spawn.h> // For mode_t
#include <stdint.h>

namespace __llvm_libc {

struct BaseSpawnFileAction {
  enum ActionType {
    OPEN = 111,
    CLOSE = 222,
    DUP2 = 333,
  };

  ActionType type;
  BaseSpawnFileAction *next;
};

struct SpawnFileOpenAction : public BaseSpawnFileAction {
  const char *path;
  int fd;
  int oflag;
  mode_t mode;
};

struct SpawnFileCloseAction : public BaseSpawnFileAction {
  int fd;
};

struct SpawnFileDup2Action : public BaseSpawnFileAction {
  int fd;
  int newfd;
};

inline void enque_spawn_action(posix_spawn_file_actions_t *actions,
                               BaseSpawnFileAction *act) {
  if (actions->__back != nullptr) {
    auto *back = reinterpret_cast<BaseSpawnFileAction *>(actions->__back);
    back->next = act;
    actions->__back = act;
  } else {
    // First action is being added.
    actions->__front = actions->__back = act;
  }
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SPAWN_FILE_ACTIONS_H
