//===-- Implementation header for rand utilities ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_RAND_UTIL_H
#define LLVM_LIBC_SRC_STDLIB_RAND_UTIL_H

namespace __llvm_libc {

extern thread_local unsigned long rand_next;

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDLIB_RAND_UTIL_H
