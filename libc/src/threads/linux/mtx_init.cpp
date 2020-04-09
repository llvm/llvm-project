//===-- Linux implementation of the mtx_init function ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/threads.h" // For mtx_t definition.
#include "src/__support/common.h"
#include "src/threads/linux/thread_utils.h"

namespace __llvm_libc {

int LLVM_LIBC_ENTRYPOINT(mtx_init)(mtx_t *mutex, int type) {
  *(reinterpret_cast<uint32_t *>(mutex->__internal_data)) = MS_Free;
  mutex->__mtx_type = type;
  return thrd_success;
}

} // namespace __llvm_libc
