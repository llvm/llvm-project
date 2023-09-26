//===-- Implementation header for strlcpy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_STRLCPY_H
#define LLVM_LIBC_SRC_STRING_STRLCPY_H

#include <string.h>

namespace LIBC_NAMESPACE {

size_t strlcpy(char *__restrict dst, const char *__restrict src, size_t size);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STRING_STRLCPY_H
