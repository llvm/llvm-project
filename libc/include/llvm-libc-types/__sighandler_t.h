//===-- Definition of struct __sighandler_t -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES___SIGHANDLER_T_H
#define LLVM_LIBC_TYPES___SIGHANDLER_T_H

#ifndef __linux__
#error "sighandler_t only available on linux"
#endif

typedef void (*__sighandler_t)(int);
typedef __sighandler_t sighandler_t;

#endif // LLVM_LIBC_TYPES___SIGHANDLER_T_H
