//===-- Implementation header for stpcpy ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_STPCPY_H
#define LLVM_LIBC_SRC_STRING_STPCPY_H

namespace LIBC_NAMESPACE {

char *stpcpy(char *__restrict dest, const char *__restrict src);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STRING_STPCPY_H
