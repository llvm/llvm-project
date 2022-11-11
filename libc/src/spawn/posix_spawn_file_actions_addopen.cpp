//===-- Implementation of posix_spawn_file_actions_addopen ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "posix_spawn_file_actions_addopen.h"

#include "file_actions.h"
#include "src/__support/common.h"

#include <errno.h>
#include <spawn.h>
#include <stdlib.h> // For malloc

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, posix_spawn_file_actions_addopen,
                   (posix_spawn_file_actions_t *__restrict actions, int fd,
                    const char *__restrict path, int oflag, mode_t mode)) {
  if (actions == nullptr)
    return EINVAL;
  if (fd < 0)
    return EBADF;

  auto *act = reinterpret_cast<SpawnFileOpenAction *>(
      malloc(sizeof(SpawnFileOpenAction)));
  if (act == nullptr)
    return ENOMEM;

  act->type = BaseSpawnFileAction::OPEN;
  act->fd = fd;
  act->path = path;
  act->oflag = oflag;
  act->mode = mode;
  act->next = nullptr;
  enque_spawn_action(actions, act);
  return 0;
}

} // namespace __llvm_libc
