//===---------- x86_64-specific implementations for pkey_{get,set}. -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYS_MMAN_LINUX_X86_64_PKEY_COMMON_H_
#define LLVM_SYS_MMAN_LINUX_X86_64_PKEY_COMMON_H_

#include <immintrin.h>

#include "hdr/errno_macros.h" // For ENOSYS
#include "hdr/stdint_proxy.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86_64)
#error "Invalid include"
#endif

namespace LIBC_NAMESPACE_DECL {
namespace pkey_common {

constexpr int KEY_COUNT = 16;
constexpr int KEY_MASK = 0x3;
constexpr int BITS_PER_KEY = 2;

// x86_64 implementation of pkey_get.
// Returns the access rights for the given pkey on success, errno otherwise.
[[gnu::target("pku")]]
LIBC_INLINE ErrorOr<int> pkey_get(int pkey) {
  if (pkey < 0 || pkey >= KEY_COUNT) {
    return Error(EINVAL);
  }

  uint32_t pkru = _rdpkru_u32();
  return (pkru >> (pkey * BITS_PER_KEY)) & KEY_MASK;
}

// x86_64 implementation of pkey_set.
// Returns 0 on success, errno otherwise.
[[gnu::target("pku")]]
LIBC_INLINE ErrorOr<int> pkey_set(int pkey, unsigned int access_rights) {
  if (pkey < 0 || pkey >= KEY_COUNT || access_rights > KEY_MASK) {
    return Error(EINVAL);
  }

  uint32_t pkru = _rdpkru_u32();
  pkru &= ~(KEY_MASK << (pkey * BITS_PER_KEY));
  pkru |= ((access_rights & KEY_MASK) << (pkey * BITS_PER_KEY));
  _wrpkru(pkru);

  return 0;
}

} // namespace pkey_common
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_SYS_MMAN_LINUX_X86_64_PKEY_COMMON_H_
