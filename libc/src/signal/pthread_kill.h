//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for pthread_kill.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SIGNAL_PTHREAD_KILL_H
#define LLVM_LIBC_SRC_SIGNAL_PTHREAD_KILL_H

#include "hdr/types/pthread_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

/// Request that a signal be delivered to a thread.
///
/// \param thread The thread to receive the signal.
/// \param sig The signal to deliver.
/// \return 0 on success, or an error number on failure.
int pthread_kill(pthread_t thread, int sig);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SIGNAL_PTHREAD_KILL_H
