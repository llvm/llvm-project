//===-- Implementation header for pthread_mutexattr_destroy -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEXATTR_DESTROY_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEXATTR_DESTROY_H

#include <pthread.h>

namespace __llvm_libc {

int pthread_mutexattr_destroy(pthread_mutexattr_t *attr);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEXATTR_DESTROY_H
