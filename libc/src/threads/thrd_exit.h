//===-- Implementation header for thrd_exit function ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_THRD_EXIT_H
#define LLVM_LIBC_SRC_THREADS_THRD_EXIT_H

#include <threads.h>

namespace LIBC_NAMESPACE {

void thrd_exit(int retval);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_THREADS_THRD_EXIT_H
