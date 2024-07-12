//===-- Implementation header of fcntl --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_FCNTL_FCNTL_H
#define LLVM_LIBC_SRC_FCNTL_FCNTL_H

namespace LIBC_NAMESPACE {

int fcntl(int fd, int cmd, ...);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_FCNTL_FCNTL_H
