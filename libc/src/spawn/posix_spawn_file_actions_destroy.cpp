//===-- Implementation of posix_spawn_file_actions_destroy ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "posix_spawn_file_actions_destroy.h"

#include "file_actions.h"
#include "src/__support/common.h"

#include <errno.h>
#include <spawn.h>
#include <stdlib.h> // For free

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, posix_spawn_file_actions_destroy,
                   (posix_spawn_file_actions_t * actions)) {
  if (actions == nullptr)
    return EINVAL;
  if (actions->__front == nullptr)
    return 0;

  auto *act = reinterpret_cast<BaseSpawnFileAction *>(actions->__front);
  actions->__front = nullptr;
  actions->__back = nullptr;

  while (act != nullptr) {
    auto *next = act->next;
    free(act);
    act = next;
  }

  return 0;
}

} // namespace __llvm_libc
