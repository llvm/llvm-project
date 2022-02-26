//===-- Linux implementation of the mtx_destroy function ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/mtx_destroy.h"
#include "include/threads.h" // For mtx_t definition.
#include "src/__support/common.h"
#include "src/__support/threads/mutex.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void, mtx_destroy, (mtx_t * mutex)) {}

} // namespace __llvm_libc
