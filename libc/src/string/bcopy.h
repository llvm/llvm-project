//===-- Implementation header for bcopy -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_BCOPY_H
#define LLVM_LIBC_SRC_STRING_BCOPY_H

#include <stddef.h> // size_t

namespace LIBC_NAMESPACE {

void bcopy(const void *src, void *dest, size_t count);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STRING_BCOPY_H
