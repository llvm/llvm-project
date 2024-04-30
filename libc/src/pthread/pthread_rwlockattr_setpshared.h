//===-- Implementation header for pthread_rwlockattr_setpshared -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_RWLOCKATTR_SETPSHARED_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_RWLOCKATTR_SETPSHARED_H

#include <pthread.h>

namespace LIBC_NAMESPACE {

int pthread_rwlockattr_setpshared(pthread_rwlockattr_t *attr, int pshared);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_RWLOCKATTR_SETPSHARED_H
