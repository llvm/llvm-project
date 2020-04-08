//===-- Implementation header for sigprocmask -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SIGNAL_SIGPROCMASK_H
#define LLVM_LIBC_SRC_SIGNAL_SIGPROCMASK_H

#include "include/signal.h"

namespace __llvm_libc {

int sigprocmask(int how, const sigset_t *__restrict set,
                sigset_t *__restrict oldset);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SIGNAL_SIGPROCMASK_H
