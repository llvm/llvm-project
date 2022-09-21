//===-- Implementation header for getuid ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_GETUID_H
#define LLVM_LIBC_SRC_UNISTD_GETUID_H

#include <unistd.h>

namespace __llvm_libc {

uid_t getuid();

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_UNISTD_GETUID_H
