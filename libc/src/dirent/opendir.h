//===-- Implementation header of opendir ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_DIRENT_OPENDIR_H
#define LLVM_LIBC_SRC_DIRENT_OPENDIR_H

#include <dirent.h>

namespace LIBC_NAMESPACE {

::DIR *opendir(const char *name);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_DIRENT_OPENDIR_H
