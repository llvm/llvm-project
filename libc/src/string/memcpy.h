//===-- Implementation header for memcpy ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMCPY_H
#define LLVM_LIBC_SRC_STRING_MEMCPY_H

#include <stddef.h> // size_t

namespace LIBC_NAMESPACE {

void *memcpy(void *__restrict, const void *__restrict, size_t);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STRING_MEMCPY_H
