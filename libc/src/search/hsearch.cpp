//===-- Implementation of hsearch -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/hsearch.h"
#include "src/__support/HashTable/table.h"
#include "src/__support/libc_assert.h"
#include "src/errno/libc_errno.h"
#include "src/search/hsearch/global.h"

namespace LIBC_NAMESPACE {
LLVM_LIBC_FUNCTION(ENTRY *, hsearch, (ENTRY item, ACTION action)) {
  ENTRY *result;
  LIBC_ASSERT(internal::global_hash_table != nullptr);
  switch (action) {
  case FIND:
    result = internal::global_hash_table->find(item.key);
    if (result == nullptr) {
      libc_errno = ESRCH;
    }
    break;
  case ENTER:
    result = internal::global_hash_table->insert(item);
    if (result == nullptr) {
      libc_errno = ENOMEM;
    }
    break;
  }
  return result;
}

} // namespace LIBC_NAMESPACE
