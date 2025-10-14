//===-- Static header for libc_constraint_handler ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ANNEX_K_LIBC_CONSTRAINT_HANDLER_H
#define LLVM_LIBC_SRC___SUPPORT_ANNEX_K_LIBC_CONSTRAINT_HANDLER_H

#include "abort_handler_s.h"
#include "hdr/types/constraint_handler_t.h"

namespace LIBC_NAMESPACE_DECL {

namespace annex_k {

LIBC_INLINE static constraint_handler_t libc_constraint_handler =
    &abort_handler_s;

} // namespace annex_k

} // namespace LIBC_NAMESPACE_DECL

#endif  // LLVM_LIBC_SRC___SUPPORT_ANNEX_K_LIBC_CONSTRAINT_HANDLER_H
