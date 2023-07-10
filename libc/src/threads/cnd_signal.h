//===-- Implementation header for cnd_signal function -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_CND_SIGNAL_H
#define LLVM_LIBC_SRC_THREADS_CND_SIGNAL_H

#include <threads.h>

namespace __llvm_libc {

int cnd_signal(cnd_t *cond);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_THREADS_CND_SIGNAL_H
