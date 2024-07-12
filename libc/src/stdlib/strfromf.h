//===-- Implementation header for strfromf ------------------------*- C++--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_STRFROMF_H
#define LLVM_LIBC_SRC_STDLIB_STRFROMF_H

#include <stddef.h>

namespace LIBC_NAMESPACE {

int strfromf(char *__restrict s, size_t n, const char *__restrict format,
             float fp);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_STRFROMF_H
