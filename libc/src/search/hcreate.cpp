//===-- Implementation of hcreate -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/hcreate.h"
#include "src/__support/HashTable/randomness.h"
#include "src/__support/HashTable/table.h"
#include "src/errno/libc_errno.h"
#include "src/search/hsearch/global.h"

namespace LIBC_NAMESPACE {
LLVM_LIBC_FUNCTION(int, hcreate, (size_t capacity)) {
  uint64_t randomness = internal::randomness::next_random_seed();
  internal::HashTable *table =
      internal::HashTable::allocate(capacity, randomness);
  if (table == nullptr) {
    libc_errno = ENOMEM;
    return 0;
  }
  internal::global_hash_table = table;
  return 1;
}

} // namespace LIBC_NAMESPACE
