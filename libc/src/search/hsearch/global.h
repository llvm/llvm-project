//===-- Global hashtable header -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SEARCH_HSEARCH_GLOBAL_H
#define LLVM_LIBC_SRC_SEARCH_HSEARCH_GLOBAL_H

namespace LIBC_NAMESPACE {
namespace internal {
extern struct HashTable *global_hash_table;
}
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SEARCH_HSEARCH_GLOBAL_H
