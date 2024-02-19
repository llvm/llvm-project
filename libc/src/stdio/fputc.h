//===-- Implementation header of fputc --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_FPUTC_H
#define LLVM_LIBC_SRC_STDIO_FPUTC_H

#include <stdio.h>

namespace LIBC_NAMESPACE {

int fputc(int c, ::FILE *stream);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDIO_FPUTC_H
