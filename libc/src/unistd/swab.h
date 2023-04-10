//===-- Implementation header for swab --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_SWAB_H
#define LLVM_LIBC_SRC_UNISTD_SWAB_H

#include <unistd.h> // For ssize_t

namespace __llvm_libc {

void swab(const void *__restrict from, void *__restrict to, ssize_t n);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_UNISTD_SWAB_H
