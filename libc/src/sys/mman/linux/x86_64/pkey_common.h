//===---------- x86_64-specific implementations for pkey_{get,set}. -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYS_MMAN_LINUX_X86_64_PKEY_COMMON_H_
#define LLVM_SYS_MMAN_LINUX_X86_64_PKEY_COMMON_H_

#include "hdr/errno_macros.h" // For ENOSYS
#include "hdr/stdint_proxy.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"

#if !defined(LIBC_TARGET_ARCH_IS_X86_64)
#error "Invalid include"
#endif

namespace LIBC_NAMESPACE_DECL {
namespace pkey_common {
namespace internal {

constexpr int MAX_KEY = 15;
constexpr int KEY_MASK = 0x3;
constexpr int BITS_PER_KEY = 2;

// This will SIGILL on CPUs that don't support PKU / OSPKE,
// but this case should never be reached as a prior pkey_alloc invocation
// would have failed more gracefully.
LIBC_INLINE uint32_t read_prku() {
  uint32_t pkru = 0;
  uint32_t edx = 0;
  asm volatile("rdpkru" : "=a"(pkru), "=d"(edx) : "c"(0));
  return pkru;
}

// This will SIGILL on CPUs that don't support PKU / OSPKE,
// but this case should never be reached as a prior pkey_alloc invocation
// would have failed more gracefully.
LIBC_INLINE void write_prku(uint32_t pkru) {
  asm volatile("wrpkru" : : "a"(pkru), "d"(0), "c"(0));
}

} // namespace internal

// x86_64 implementation of pkey_get.
// Returns the access rights for the given pkey on success, errno otherwise.
LIBC_INLINE ErrorOr<int> pkey_get(int pkey) {
  if (pkey < 0 || pkey > internal::MAX_KEY) {
    return Error(EINVAL);
  }

  uint32_t pkru = internal::read_prku();
  return (pkru >> (pkey * internal::BITS_PER_KEY)) & internal::KEY_MASK;
}

// x86_64 implementation of pkey_set.
// Returns 0 on success, errno otherwise.
LIBC_INLINE ErrorOr<int> pkey_set(int pkey, unsigned int access_rights) {
  if (pkey < 0 || pkey > internal::MAX_KEY ||
      access_rights > internal::KEY_MASK) {
    return Error(EINVAL);
  }

  uint32_t pkru = internal::read_prku();
  pkru &= ~(internal::KEY_MASK << (pkey * internal::BITS_PER_KEY));
  pkru |=
      ((access_rights & internal::KEY_MASK) << (pkey * internal::BITS_PER_KEY));
  internal::write_prku(pkru);

  return 0;
}

} // namespace pkey_common
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_SYS_MMAN_LINUX_X86_64_PKEY_COMMON_H_
