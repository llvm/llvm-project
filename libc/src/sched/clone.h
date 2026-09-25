//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for clone.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SCHED_CLONE_H
#define LLVM_LIBC_SRC_SCHED_CLONE_H

#include "hdr/types/pid_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

/// Creates a new child process with execution starting at \p func(\p arg).
///
/// \param func The child process entry function. Must not be null.
/// \param stack Pointer to the top of the memory allocated for the child stack.
///              Must not be null.
/// \param flags Sharing flags (CLONE_*) and exit signal for the parent.
/// \param arg Argument passed to \p func.
/// \param ... Positional arguments depending on \p flags, in order:
///            - pid_t *parent_tid if CLONE_PARENT_SETTID or CLONE_PIDFD is set.
///            - void *tls if CLONE_SETTLS is set.
///            - pid_t *child_tid if CLONE_CHILD_SETTID or CLONE_CHILD_CLEARTID
///              is set.
///            Preceding arguments must be provided if a subsequent argument is
///            needed.
/// \return On success, returns the process ID of the child. On failure, returns
///         -1 and sets errno.
int clone(int (*func)(void *), void *stack, int flags, void *arg, ...);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SCHED_CLONE_H
