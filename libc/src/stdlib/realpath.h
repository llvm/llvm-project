//===-- Implementation header for realpath ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_REALPATH_H
#define LLVM_LIBC_SRC_STDLIB_REALPATH_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

char *realpath(const char *path, char *resolved_path);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_REALPATH_H
