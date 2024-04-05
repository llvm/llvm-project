//===-- Implementation header of fseeko -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_FSEEKO_H
#define LLVM_LIBC_SRC_STDIO_FSEEKO_H

#include <stdio.h>

namespace LIBC_NAMESPACE {

int fseeko(::FILE *stream, off_t offset, int whence);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDIO_FSEEKO_H
