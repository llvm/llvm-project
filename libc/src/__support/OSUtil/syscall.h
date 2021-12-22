//===--------------- Internal syscall declarations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_OSUTIL_SYSCALL_H
#define LLVM_LIBC_SRC_SUPPORT_OSUTIL_SYSCALL_H

#ifdef __unix__
#include "linux/syscall.h"
#endif

#endif // LLVM_LIBC_SRC_SUPPORT_OSUTIL_SYSCALL_H
