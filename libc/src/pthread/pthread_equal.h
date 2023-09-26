//===-- Implementation header for pthread_equal function -----------*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_EQUAL_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_EQUAL_H

#include <pthread.h>

namespace LIBC_NAMESPACE {

int pthread_equal(pthread_t lhs, pthread_t rhs);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_EQUAL_H
