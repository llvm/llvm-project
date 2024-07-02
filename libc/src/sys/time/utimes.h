//===-- Implementation header for socket ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_TIME_UTIMES_H
#define LLVM_LIBC_SRC_SYS_TIME_UTIMES_H

namespace LIBC_NAMESPACE {

int utimes(const char *pathname, const struct timeval times[2]);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_TIME_UTIMES_H
