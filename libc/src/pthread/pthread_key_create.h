//===-- Implementation header for pthread_key_create ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_KEY_CREATE_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_KEY_CREATE_H

#include <pthread.h>

namespace __llvm_libc {

int pthread_key_create(pthread_key_t *key, __pthread_tss_dtor_t dtor);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_KEY_CREATE_H
