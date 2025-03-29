//===-- Implementation of posix_spawn_file_actions_destroy ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "posix_spawn_file_actions_destroy.h"

#include "file_actions.h"

#include "src/__support/CPP/new.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"

#include <spawn.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, posix_spawn_file_actions_destroy,
                   (posix_spawn_file_actions_t * actions)) {
  if (actions == nullptr)
    return EINVAL;
  if (actions->__front == nullptr)
    return 0;

  auto *act = reinterpret_cast<BaseSpawnFileAction *>(actions->__front);
  actions->__front = nullptr;
  actions->__back = nullptr;
  if (act == nullptr)
    return 0;

  while (act != nullptr) {
    auto *temp = act;
    act = act->next;
    switch (temp->type) {
    case BaseSpawnFileAction::OPEN:
      delete reinterpret_cast<SpawnFileOpenAction *>(temp);
      break;
    case BaseSpawnFileAction::CLOSE:
      delete reinterpret_cast<SpawnFileCloseAction *>(temp);
      break;
    case BaseSpawnFileAction::DUP2:
      delete reinterpret_cast<SpawnFileDup2Action *>(temp);
      break;
    }
  }

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
