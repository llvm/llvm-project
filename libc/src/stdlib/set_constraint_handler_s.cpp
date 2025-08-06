//===-- Implementation of set_constraint_handler_s ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "set_constraint_handler_s.h"
#include "src/__support/annex_k/abort_handler_s.h"
#include "src/__support/annex_k/libc_constraint_hander.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(constraint_handler_t, set_constraint_handler_s,
                   (constraint_handler_t handler)) {
  constraint_handler_t previous_handler = libc_constraint_handler;

  if (!handler) {
    libc_constraint_handler = &abort_handler_s;
  } else {
    libc_constraint_handler = handler;
  }

  return previous_handler;
}

} // namespace LIBC_NAMESPACE_DECL
