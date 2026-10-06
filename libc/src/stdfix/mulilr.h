//===-- Implementation header for mulilr -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDFIX_MULILR_H
#define LLVM_LIBC_SRC_STDFIX_MULILR_H

#include "include/llvm-libc-macros/stdfix-macros.h" // long fract
#include "src/__support/macros/config.h"            // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

long int mulilr(long fract f, long int i);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDFIX_MULILR_H
