//===-- Linux implementation of the pthread_atfork function ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_atfork.h"

#include "src/__support/common.h"
#include "src/__support/fork_callbacks.h"

#include <errno.h>
#include <pthread.h> // For pthread_* type definitions.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, pthread_atfork,
                   (__atfork_callback_t prepare, __atfork_callback_t parent,
                    __atfork_callback_t child)) {
  return register_atfork_callbacks(prepare, parent, child) ? 0 : ENOMEM;
}

} // namespace __llvm_libc
