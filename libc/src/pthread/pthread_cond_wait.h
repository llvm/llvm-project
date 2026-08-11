//===-- Implementation header for pthread_cond_wait ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_COND_WAIT_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_COND_WAIT_H

#include "include/llvm-libc-types/pthread_cond_t.h"
#include "include/llvm-libc-types/pthread_mutex_t.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

int pthread_cond_wait(pthread_cond_t *__restrict cond,
                      pthread_mutex_t *__restrict mutex);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_COND_WAIT_H
