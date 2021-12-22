//===----------------------- Linux syscalls ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_SYSCALL_H
#define LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_SYSCALL_H

#include "src/__support/architectures.h"

#ifdef LLVM_LIBC_ARCH_X86_64
#include "x86_64/syscall.h"
#endif

#endif // LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_SYSCALL_H
