//===-- Global hashtable implementation -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {
struct HashTable *global_hash_table = nullptr;
}
} // namespace LIBC_NAMESPACE_DECL
