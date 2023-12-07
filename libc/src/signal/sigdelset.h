//===-- Implementation header for sigdelset ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SIGNAL_SIGDELSET_H
#define LLVM_LIBC_SRC_SIGNAL_SIGDELSET_H

#include <signal.h>

namespace __llvm_libc {

int sigdelset(sigset_t *set, int signum);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SIGNAL_SIGDELSET_H
