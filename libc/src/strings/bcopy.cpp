//===-- Implementation of bcopy -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/strings/bcopy.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memmove.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, bcopy, (const void *src, void *dst, size_t count)) {
  return inline_memmove(dst, src, count);
}

} // namespace LIBC_NAMESPACE_DECL
