//===-- Implementation header of fgets --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_FGETS_H
#define LLVM_LIBC_SRC_STDIO_FGETS_H

#include <stdio.h>

namespace LIBC_NAMESPACE {

char *fgets(char *__restrict str, int count, ::FILE *__restrict raw_stream);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDIO_FGETS_H
