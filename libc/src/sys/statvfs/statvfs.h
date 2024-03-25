//===-- Implementation header for statvfs -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_STATVFS_STATVFS_H
#define LLVM_LIBC_SRC_SYS_STATVFS_STATVFS_H

#include "llvm-libc-types/struct_statvfs.h"

namespace LIBC_NAMESPACE {

int statvfs(const char *__restrict path, struct statvfs *__restrict buf);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_STATVFS_STATVFS_H
