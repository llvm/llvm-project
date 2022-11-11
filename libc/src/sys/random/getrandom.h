//===-- Implementation header for getrandom ----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_RANDOM_GETRANDOM_H
#define LLVM_LIBC_SRC_SYS_RANDOM_GETRANDOM_H

#include <sys/random.h>

namespace __llvm_libc {

ssize_t getrandom(void *buf, size_t buflen, unsigned int flags);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SYS_RANDOM_GETRANDOM_H
