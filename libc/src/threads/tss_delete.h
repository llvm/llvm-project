//===-- Implementation header for tss_delete --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_TSS_DELETE_H
#define LLVM_LIBC_SRC_THREADS_TSS_DELETE_H

#include <threads.h>

namespace __llvm_libc {

void tss_delete(tss_t key);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_THREADS_TSS_DELETE_H
