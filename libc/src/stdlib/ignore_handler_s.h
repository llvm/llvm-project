//===-- Implementation header for ignore_handler_s --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_IGNORE_HANDLER_S_H
#define LLVM_LIBC_SRC_STDLIB_IGNORE_HANDLER_S_H

#include "hdr/types/errno_t.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

void ignore_handler_s(const char *__restrict msg, void *__restrict ptr,
                      errno_t error);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_IGNORE_HANDLER_S_H
