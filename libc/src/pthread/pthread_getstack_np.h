//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for pthread_getstack_np (LLVM-libc extension).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETSTACK_NP_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETSTACK_NP_H

#include "hdr/types/pthread_t.h"
#include "hdr/types/size_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// Returns the stack address and stack size of the thread referred to by the
// thread ID `th` in the buffers pointed to by `stackaddr` and `stacksize`,
// respectively. `stackaddr` points to the lowest address in the region,
// regardless of the direction of stack growth.
//
// If the size of the stack is not fixed (stack can grow dynamically), the
// stacksize argument is set to PTHREAD_STACK_DYNAMIC_NP (0) and stackaddr
// points to the base of the stack (as if it was describing the empty stack
// range after popping its entire contents). This allows one to compute the
// currently used part of the stack by taking the range between this value and
// the current stack pointer. This behavior is specific to the main thread.
//
// On success, the function returns zero; on error, it returns a nonzero error
// number. Currently, the function always succeeds.
//
// This function is async-signal-safe.
int pthread_getstack_np(pthread_t th, void **__restrict stackaddr,
                        size_t *__restrict stacksize);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETSTACK_NP_H
