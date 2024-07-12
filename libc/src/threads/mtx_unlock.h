//===-- Implementation header for mtx_unlock function -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_MTX_UNLOCK_H
#define LLVM_LIBC_SRC_THREADS_MTX_UNLOCK_H

#include <threads.h>

namespace LIBC_NAMESPACE {

int mtx_unlock(mtx_t *mutex);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_THREADS_MTX_UNLOCK_H
