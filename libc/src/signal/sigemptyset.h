//===-- Implementation header for sigemptyset -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SIGNAL_SIGEMPTYSET_H
#define LLVM_LIBC_SRC_SIGNAL_SIGEMPTYSET_H

#include <signal.h>

namespace __llvm_libc {

int sigemptyset(sigset_t *set);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SIGNAL_SIGEMPTYSET_H
