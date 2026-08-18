//===-- Implementation header for divir -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDFIX_DIVIR_H
#define LLVM_LIBC_SRC_STDFIX_DIVIR_H

#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int divir(int n, fract d);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDFIX_DIVIR_H
