//===-- C standard library header errno.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_ERRNO_H
#define LLVM_LIBC_ERRNO_H

#include "__llvm-libc-common.h"

#ifdef __linux__

#include <linux/errno.h>

#ifndef ENOTSUP
#define ENOTSUP EOPNOTSUPP
#endif // ENOTSUP

#include "llvm-libc-macros/linux/error-number-macros.h"

#else // __linux__
#include "llvm-libc-macros/generic-error-number-macros.h"
#endif

#if !defined(__AMDGPU__) && !defined(__NVPTX__)

#ifdef __cplusplus
extern "C" {
extern thread_local int __llvmlibc_errno;
}
#else
extern _Thread_local int __llvmlibc_errno;
#endif // __cplusplus

#define errno __llvmlibc_errno
#endif

#endif // LLVM_LIBC_ERRNO_H
