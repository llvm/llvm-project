//===-- Definitions of common POSIX types ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This header file does not have a header guard. It is internal to LLVM libc
// and intended to be used to pick specific definitions without polluting the
// public headers with unnecessary definitions.

#if defined(__need_off_t) && !defined(__llvm_libc_off_t_defined)
typedef __INT64_TYPE__ off_t;
#define __llvm_libc_off_t_defined
#endif // __need_off_t

#if defined(__need_ssize_t) && !defined(__llvm_libc_ssize_t_defined)
typedef __INT64_TYPE__ ssize_t;
#define __llvm_libc_ssize_t_defined
#endif // __need_ssize_t
