//===-- Implementation header for pthread_getname_np function ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETNAME_NP_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETNAME_NP_H

#include <pthread.h>
#include <stddef.h>

namespace __llvm_libc {

int pthread_getname_np(pthread_t, char *, size_t);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETNAME_NP_H
