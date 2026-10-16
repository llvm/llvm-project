//===-- Implementation header for pthread_mutex_trylock function ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEX_TRYLOCK_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEX_TRYLOCK_H

#include "include/llvm-libc-types/pthread_mutex_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int pthread_mutex_trylock(pthread_mutex_t *mutex);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEX_TRYLOCK_H
