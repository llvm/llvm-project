//===-- Implementation header for mtx_init function -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_MTX_INIT_H
#define LLVM_LIBC_SRC_THREADS_MTX_INIT_H

#include <threads.h>

namespace LIBC_NAMESPACE {

int mtx_init(mtx_t *mutex, int type);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_THREADS_MTX_INIT_H
