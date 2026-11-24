//===-- Implementation of posix_spawn_file_actions_destroy ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "posix_spawn_file_actions_destroy.h"

#include "file_actions.h"

#include "hdr/errno_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

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
    // A delete expression would need a virtual destructor, but the actions are
    // trivially destructible (add_action checks that), so we can just free the
    // memory.
    BaseSpawnFileAction::operator delete(temp);
  }

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
