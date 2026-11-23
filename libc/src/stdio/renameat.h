//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header of renameat.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_RENAMEAT_H
#define LLVM_LIBC_SRC_STDIO_RENAMEAT_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int renameat(int olddirfd, const char *oldpath, int newdirfd,
             const char *newpath);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_RENAMEAT_H
