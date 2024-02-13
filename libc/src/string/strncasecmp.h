//===-- Implementation header for strcasecmp --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_STRNCASECMP_H
#define LLVM_LIBC_SRC_STRING_STRNCASECMP_H

#include <stddef.h>

namespace LIBC_NAMESPACE {

int strncasecmp(const char *left, const char *right, size_t n);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STRING_STRNCASECMP_H
