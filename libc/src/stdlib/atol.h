//===-- Implementation header for atol --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_ATOL_H
#define LLVM_LIBC_SRC_STDLIB_ATOL_H

namespace LIBC_NAMESPACE {

long atol(const char *str);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDLIB_ATOL_H
