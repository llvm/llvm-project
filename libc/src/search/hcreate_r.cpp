//===-- Implementation of hcreate_r -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/hcreate_r.h"
#include "src/__support/HashTable/randomness.h"
#include "src/__support/HashTable/table.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(int, hcreate_r,
                   (size_t capacity, struct hsearch_data *htab)) {
  if (htab == nullptr) {
    libc_errno = EINVAL;
    return 0;
  }
  uint64_t randomness = internal::randomness::next_random_seed();
  internal::HashTable *table =
      internal::HashTable::allocate(capacity, randomness);
  if (table == nullptr) {
    libc_errno = ENOMEM;
    return 0;
  }
  htab->__opaque = table;
  return 1;
}

} // namespace LIBC_NAMESPACE_DECL
