//===-- Implementation header for tss_set function --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_TSS_SET_H
#define LLVM_LIBC_SRC_THREADS_TSS_SET_H

#include <threads.h>

namespace __llvm_libc {

int tss_set(tss_t, void *);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_THREADS_TSS_SET_H
