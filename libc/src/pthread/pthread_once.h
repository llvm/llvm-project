//===-- Implementation header for pthread_once function ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_PTHREAD_ONCE_H
#define LLVM_LIBC_SRC_THREADS_PTHREAD_ONCE_H

#include <pthread.h>

namespace __llvm_libc {

int pthread_once(pthread_once_t *flag, __pthread_once_func_t func);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_THREADS_PTHREAD_ONCE_H
