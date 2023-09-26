//===-- Implementation header for malloc ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>

#ifndef LLVM_LIBC_SRC_STDLIB_MALLOC_H
#define LLVM_LIBC_SRC_STDLIB_MALLOC_H

namespace LIBC_NAMESPACE {

void *malloc(size_t size);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_MALLOC_H
