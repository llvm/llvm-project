//===-- Definition of pthread_once_t type ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_PTHREAD_ONCE_T_H
#define LLVM_LIBC_TYPES_PTHREAD_ONCE_T_H

// An int, as it is in glibc, musl and bionic, so that code written against them
// can initialise and compare it as one.
typedef int pthread_once_t;

#endif // LLVM_LIBC_TYPES_PTHREAD_ONCE_T_H
