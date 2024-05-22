//===-- Global hashtable implementation -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace LIBC_NAMESPACE {
namespace internal {
struct HashTable *global_hash_table = nullptr;
}
} // namespace LIBC_NAMESPACE
