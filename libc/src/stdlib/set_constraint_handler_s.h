//===-- Implementation header for set_constraint_handler_s ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_SET_CONSTRAINT_HANDLER_S_H
#define LLVM_LIBC_SRC_STDLIB_SET_CONSTRAINT_HANDLER_S_H

#include "hdr/types/constraint_handler_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

constraint_handler_t set_constraint_handler_s(constraint_handler_t handler);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_SET_CONSTRAINT_HANDLER_S_H
