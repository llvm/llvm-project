//===-- Implementation header of dirfd --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_DIRENT_DIRFD_H
#define LLVM_LIBC_SRC_DIRENT_DIRFD_H

#include <dirent.h>

namespace LIBC_NAMESPACE {

int dirfd(::DIR *dir);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_DIRENT_DIRFD_H
