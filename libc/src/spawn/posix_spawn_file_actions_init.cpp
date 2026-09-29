//===-- Implementation of posix_spawn_file_actions_init -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "posix_spawn_file_actions_init.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include <spawn.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, posix_spawn_file_actions_init,
                   (posix_spawn_file_actions_t * actions)) {
  actions->__front = nullptr;
  actions->__back = nullptr;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
