//===-- Implementation header for sigaltstack -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SIGNAL_SIGALSTACK_H
#define LLVM_LIBC_SRC_SIGNAL_SIGALSTACK_H

#include <signal.h>

namespace __llvm_libc {

int sigaltstack(const stack_t *__restrict ss, stack_t *__restrict oss);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SIGNAL_SIGALSTACK_H
