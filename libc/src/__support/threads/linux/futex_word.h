//===--- Definition of a type for a futex word ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_THREADS_LINUX_FUTEX_WORD_H
#define LLVM_LIBC_SRC_SUPPORT_THREADS_LINUX_FUTEX_WORD_H

#include "src/__support/architectures.h"

namespace __llvm_libc {

#if defined(LLVM_LIBC_ARCH_X86_64) || defined(LLVM_LIBC_ARCH_AARCH64)
using FutexWordType = unsigned int;
#else
#error "FutexWordType not defined for the target architecture."
#endif

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_THREADS_LINUX_FUTEX_WORD_H
