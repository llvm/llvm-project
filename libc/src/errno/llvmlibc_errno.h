//===-- Implementation header for errno -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_ERRNO_LLVMLIBC_ERRNO_H
#define LLVM_LIBC_SRC_ERRNO_LLVMLIBC_ERRNO_H

// Internal code should use this and not use the errno macro from the
// public header.
extern thread_local int __llvmlibc_errno;
#define llvmlibc_errno __llvmlibc_errno

#endif // LLVM_LIBC_SRC_ERRNO_LLVMLIBC_ERRNO_H
