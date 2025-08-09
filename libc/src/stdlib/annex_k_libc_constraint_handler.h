//===-- Static header for libc_constraint_handler ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_ANNEX_K_LIBC_CONSTRAINT_HANDLER_H
#define LLVM_LIBC_SRC_STDLIB_ANNEX_K_LIBC_CONSTRAINT_HANDLER_H

#include "hdr/types/constraint_handler_t.h"
#include "src/__support/common.h"
#include "src/stdlib/abort_handler_s.h"

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE static constraint_handler_t libc_constraint_handler =
    &abort_handler_s;

}

#endif // LLVM_LIBC_SRC_STDLIB_ANNEX_K_LIBC_CONSTRAINT_HANDLER_H
