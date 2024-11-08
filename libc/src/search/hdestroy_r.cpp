//===-- Implementation of hdestroy_r ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/hdestroy_r.h"
#include "src/__support/HashTable/table.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(void, hdestroy_r, (struct hsearch_data * htab)) {
  if (htab == nullptr) {
    libc_errno = EINVAL;
    return;
  }
  internal::HashTable *table =
      static_cast<internal::HashTable *>(htab->__opaque);
  internal::HashTable::deallocate(table);
  htab->__opaque = nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
