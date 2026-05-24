//===-- Implementation of hsearch_r -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/hsearch_r.h"
#include "src/__support/HashTable/table.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(int, hsearch_r,
                   (ENTRY item, ACTION action, ENTRY **retval,
                    struct hsearch_data *htab)) {
  if (htab == nullptr) {
    libc_errno = EINVAL;
    return 0;
  }
  internal::HashTable *table =
      static_cast<internal::HashTable *>(htab->__opaque);
  switch (action) {
  case FIND:
    *retval = table->find(item.key);
    if (*retval == nullptr) {
      libc_errno = ESRCH;
      return 0;
    }
    break;
  case ENTER:
    *retval = internal::HashTable::insert(table, item);
    htab->__opaque = table;
    if (*retval == nullptr) {
      libc_errno = ENOMEM;
      return 0;
    }
    break;
  }
  return 1;
}

} // namespace LIBC_NAMESPACE_DECL
