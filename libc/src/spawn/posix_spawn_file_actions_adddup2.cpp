//===-- Implementation of posix_spawn_file_actions_adddup2 ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "posix_spawn_file_actions_adddup2.h"

#include "file_actions.h"
#include "src/__support/common.h"

#include <errno.h>
#include <spawn.h>
#include <stdlib.h> // For malloc

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, posix_spawn_file_actions_adddup2,
                   (posix_spawn_file_actions_t * actions, int fd, int newfd)) {
  if (actions == nullptr)
    return EINVAL;
  if (fd < 0 || newfd < 0)
    return EBADF;

  auto *act = reinterpret_cast<SpawnFileDup2Action *>(
      malloc(sizeof(SpawnFileDup2Action)));
  if (act == nullptr)
    return ENOMEM;

  act->type = BaseSpawnFileAction::DUP2;
  act->fd = fd;
  act->newfd = newfd;
  act->next = nullptr;
  enque_spawn_action(actions, act);
  return 0;
}

} // namespace __llvm_libc
