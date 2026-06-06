//===-- Implementation of catopen -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/nl_types/catopen.h"
#include "include/llvm-libc-types/nl_catd.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(nl_catd, catopen,
                   ([[maybe_unused]] const char *name,
                    [[maybe_unused]] int flag)) {
  // TODO: Add implementation for message catalogs. For now, return error
  // regardless of input.
  libc_errno = EINVAL;
  return reinterpret_cast<nl_catd>(-1);
}

} // namespace LIBC_NAMESPACE_DECL
