//===-- Implementation header for divilr ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDFIX_DIVILR_H
#define LLVM_LIBC_SRC_STDFIX_DIVILR_H

#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

long int divilr(long int n, long fract d);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDFIX_DIVILR_H
