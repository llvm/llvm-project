//===-- Implementation header for cnd_wait function -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_CND_WAIT_H
#define LLVM_LIBC_SRC_THREADS_CND_WAIT_H

#include <threads.h>

namespace __llvm_libc {

int cnd_wait(cnd_t *cond, mtx_t *mutex);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_THREADS_CND_WAIT_H
