//===-- Implementation header for pthread_mutexattr_setrobust ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEXATTR_SETROBUST_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEXATTR_SETROBUST_H

#include <pthread.h>

namespace LIBC_NAMESPACE {

int pthread_mutexattr_setrobust(pthread_mutexattr_t *__restrict attr,
                                int robust);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEXATTR_SETROBUST_H
