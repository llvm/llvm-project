//===-- Implementation header for catopen -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_NL_TYPES_CATOPEN_H
#define LLVM_LIBC_SRC_NL_TYPES_CATOPEN_H

#include "include/llvm-libc-types/nl_catd.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

nl_catd catopen(const char *name, int flag);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_NL_TYPES_CATOPEN_H
