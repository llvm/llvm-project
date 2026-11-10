//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal definitions for pthread_attr_t.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_ATTR_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_ATTR_H

#include "hdr/pthread_macros.h"
#include "hdr/sched_macros.h"
#include "hdr/types/pthread_attr_t.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/thread.h"

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE_VAR constexpr pthread_attr_t DEFAULT_PTHREAD_ATTR = {
    PTHREAD_CREATE_JOINABLE,   // Not detached
    PTHREAD_INHERIT_SCHED,     // Default inherit scheduler
    SCHED_OTHER,               // Default scheduling policy
    {},                        // Default scheduling parameters
    nullptr,                   // Let the thread manage its stack
    Thread::DEFAULT_STACKSIZE, // stack size.
    Thread::DEFAULT_GUARDSIZE, // Default page size for the guard size.
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_ATTR_H
