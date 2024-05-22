//===-- Implementation header for strerror_r --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_STRERROR_R_H
#define LLVM_LIBC_SRC_STRING_STRERROR_R_H

#include <stddef.h>

namespace LIBC_NAMESPACE {

char *strerror_r(int err_num, char *buf, size_t buflen);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STRING_STRERROR_R_H
