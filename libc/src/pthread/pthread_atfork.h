//===-- Implementation header for pthread_atfork function -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_ATFORK_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_ATFORK_H

#include <pthread.h>

namespace __llvm_libc {

int pthread_atfork(__atfork_callback_t prepare, __atfork_callback_t parent,
                   __atfork_callback_t child);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_ATFORK_H
