//===-- Implementation of hdestroy ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/hdestroy.h"
#include "src/__support/HashTable/table.h"
#include "src/__support/macros/config.h"
#include "src/search/hsearch/global.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(void, hdestroy, (void)) {
  // HashTable::deallocate will check for nullptr. It will be a no-op if
  // global_hash_table is null.
  internal::HashTable::deallocate(internal::global_hash_table);
  internal::global_hash_table = nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
