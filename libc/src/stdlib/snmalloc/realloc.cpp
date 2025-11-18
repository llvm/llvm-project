//===-- Implementation of realloc -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/snmalloc/override.h"

#include "snmalloc/snmalloc.h"
#include "src/__support/common.h"
#include "src/stdlib/realloc.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(void *, realloc, (void *ptr, size_t size)) {
  return snmalloc::libc::realloc(ptr, size);
}
} // namespace LIBC_NAMESPACE_DECL
