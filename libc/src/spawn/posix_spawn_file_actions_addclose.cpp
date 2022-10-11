//===-- Implementation of posix_spawn_file_actions_addclose ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "posix_spawn_file_actions_addclose.h"

#include "file_actions.h"
#include "src/__support/common.h"

#include <errno.h>
#include <spawn.h>
#include <stdlib.h> // For malloc

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, posix_spawn_file_actions_addclose,
                   (posix_spawn_file_actions_t *__restrict actions, int fd)) {
  if (actions == nullptr)
    return EINVAL;
  if (fd < 0)
    return EBADF;

  auto *act = reinterpret_cast<SpawnFileCloseAction *>(
      malloc(sizeof(SpawnFileCloseAction)));
  if (act == nullptr)
    return ENOMEM;

  act->type = BaseSpawnFileAction::CLOSE;
  act->fd = fd;
  enque_spawn_action(actions, act);
  return 0;
}

} // namespace __llvm_libc
