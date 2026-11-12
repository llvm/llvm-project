//===-- Implementation header for pthread_barrier_destroy --------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_BARRIER_DESTROY_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_BARRIER_DESTROY_H

#include "hdr/types/pthread_barrier_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int pthread_barrier_destroy(pthread_barrier_t *b);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_BARRIER_DESTROY_H
