//===-- Implementation header for thrd_current function ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_THRD_CURRENT_H
#define LLVM_LIBC_SRC_THREADS_THRD_CURRENT_H

#include "include/threads.h"

namespace __llvm_libc {

thrd_t thrd_current();

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_THREADS_THRD_CURRENT_H
