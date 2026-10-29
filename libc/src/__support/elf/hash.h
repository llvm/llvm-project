//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// ELF symbol hash functions
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_HASH_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_HASH_H

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// The hash used by DT_HASH, defined by the System V ABI. The result is 28
// bits: the top nibble is folded back in and then cleared on every round.
LIBC_INLINE constexpr uint32_t sysv_hash(const char *name) {
  uint32_t h = 0;
  for (; *name != '\0'; ++name) {
    h = (h << 4) + static_cast<unsigned char>(*name);
    uint32_t high = h & 0xf0000000u;
    if (high != 0)
      h ^= high >> 24;
    h &= ~high;
  }
  return h;
}

// The hash used by DT_GNU_HASH, a djb2 variant.
LIBC_INLINE constexpr uint32_t gnu_hash(const char *name) {
  uint32_t h = 5381;
  for (; *name != '\0'; ++name)
    h = h * 33 + static_cast<unsigned char>(*name);
  return h;
}

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_HASH_H
