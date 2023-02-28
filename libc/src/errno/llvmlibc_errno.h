//===-- Implementation header for errno -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_ERRNO_LLVMLIBC_ERRNO_H
#define LLVM_LIBC_SRC_ERRNO_LLVMLIBC_ERRNO_H

#include <errno.h>

// DEPRECATED: Use libc_errno from libc_errno.h instead. This macro is only
// present to facilitate gradual transition (as in, in multiple simple patches)
// to libc_errno.
// TODO: After all of libc/src and libc/test is switched over to use libc_errno,
// remove this macro and header file.
#define llvmlibc_errno errno

#endif // LLVM_LIBC_SRC_ERRNO_LLVMLIBC_ERRNO_H
