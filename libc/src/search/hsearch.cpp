//===-- Implementation of hsearch -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/hsearch.h"
#include "src/__support/HashTable/randomness.h"
#include "src/__support/HashTable/table.h"
#include "src/errno/libc_errno.h"
#include "src/search/hsearch/global.h"

namespace LIBC_NAMESPACE {
LLVM_LIBC_FUNCTION(ENTRY *, hsearch, (ENTRY item, ACTION action)) {
  ENTRY *result;
  if (internal::global_hash_table == nullptr) {
    // If global_hash_table is null, we create a new hash table with a minimal
    // capacity. Such hashtable will be expanded as needed.
    uint64_t randomness = internal::randomness::next_random_seed();
    internal::global_hash_table = internal::HashTable::allocate(0, randomness);
  }

  // In rare cases, the global hashtable may still fail to allocate. We treat it
  // as ESRCH or ENOMEM depending on the action.
  switch (action) {
  case FIND:
    result = internal::global_hash_table
                 ? internal::global_hash_table->find(item.key)
                 : nullptr;
    if (result == nullptr) {
      libc_errno = ESRCH;
    }
    break;
  case ENTER:
    result =
        internal::global_hash_table
            ? internal::HashTable::insert(internal::global_hash_table, item)
            : nullptr;
    if (result == nullptr) {
      libc_errno = ENOMEM;
    }
    break;
  }
  return result;
}

} // namespace LIBC_NAMESPACE
