//===-- Implementation of catclose ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/nl_types/catclose.h"
#include "include/llvm-libc-types/nl_catd.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, catclose, ([[maybe_unused]] nl_catd catalog)) {
  // TODO: Add implementation for message catalogs. For now, return error
  // regardless of input.
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
