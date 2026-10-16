//===-- Implementation of the pthread_barrier_init function ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_barrier_init.h"

#include "hdr/types/pthread_barrier_t.h"
#include "hdr/types/pthread_barrierattr_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/linux/barrier.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_barrier_init,
                   (pthread_barrier_t * b,
                    const pthread_barrierattr_t *__restrict attr,
                    unsigned count)) {
  return Barrier::init(reinterpret_cast<Barrier *>(b), attr, count);
}

} // namespace LIBC_NAMESPACE_DECL
