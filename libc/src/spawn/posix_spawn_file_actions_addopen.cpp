//===-- Implementation of posix_spawn_file_actions_addopen ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "posix_spawn_file_actions_addopen.h"

#include "file_actions.h"
#include "src/__support/CPP/new.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"

#include <spawn.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, posix_spawn_file_actions_addopen,
                   (posix_spawn_file_actions_t *__restrict actions, int fd,
                    const char *__restrict path, int oflag, mode_t mode)) {
  if (actions == nullptr)
    return EINVAL;
  if (fd < 0)
    return EBADF;

  AllocChecker ac;
  auto *act = new (ac) SpawnFileOpenAction(path, fd, oflag, mode);
  if (act == nullptr)
    return ENOMEM;
  BaseSpawnFileAction::add_action(actions, act);

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
