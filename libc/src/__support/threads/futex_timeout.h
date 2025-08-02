//===--- Futex With Timeout Support -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_SUPPORT_THREADS_FUTEX_TIMEOUT_H
#define LLVM_LIBC_SRC_SUPPORT_THREADS_FUTEX_TIMEOUT_H

// TODO: implement futex for other platforms.
#ifdef __linux__
#include "src/__support/threads/linux/futex_timeout.h"
#else
#error "Unsupported platform"
#endif

#endif // LLVM_LIBC_SRC_SUPPORT_THREADS_FUTEX_TIMEOUT_H
