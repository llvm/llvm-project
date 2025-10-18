//===-- Implementation for abort_handler_s ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/abort_handler_s.h"
#include "src/__support/annex_k/abort_handler_s.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, abort_handler_s,
                   (const char *__restrict msg, void *__restrict ptr,
                    errno_t error)) {
  return annex_k::abort_handler_s(msg, ptr, error);
}

} // namespace LIBC_NAMESPACE_DECL
