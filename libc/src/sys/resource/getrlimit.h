//===-- Implementation header for getrlimit ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_STAT_GETRLIMIT_H
#define LLVM_LIBC_SRC_SYS_STAT_GETRLIMIT_H

#include <sys/resource.h>

namespace __llvm_libc {

int getrlimit(int resource, struct rlimit *lim);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SYS_STAT_GETRLIMIT_H
