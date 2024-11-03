//===-- Implementation header for dup3 --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_DUP3_H
#define LLVM_LIBC_SRC_UNISTD_DUP3_H

#include <unistd.h>

namespace LIBC_NAMESPACE {

int dup3(int oldfd, int newfd, int flags);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_UNISTD_DUP3_H
