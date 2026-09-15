//===-- Implementation header for setcontext --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UCONTEXT_SETCONTEXT_H
#define LLVM_LIBC_SRC_UCONTEXT_SETCONTEXT_H

#include "include/llvm-libc-types/ucontext_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int setcontext(const ucontext_t *ucp) noexcept;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_UCONTEXT_SETCONTEXT_H
