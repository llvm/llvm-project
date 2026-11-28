//===-- Implementation of bcmp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/strings/bcmp.h"
#include "src/__support/common.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/string/memory_utils/inline_bcmp.h"

namespace LIBC_NAMESPACE_DECL {

// Align to a 64-byte cache line on 64-bit targets so the 4..8 byte fast path
// fits within a single cache line and avoids alignment-sensitive latency
// swings.
#if __has_cpp_attribute(gnu::aligned) &&                                       \
    (defined(LIBC_TARGET_ARCH_IS_X86_64) ||                                    \
     defined(LIBC_TARGET_ARCH_IS_AARCH64) ||                                   \
     defined(LIBC_TARGET_ARCH_IS_RISCV64))
[[gnu::aligned(64)]]
#endif
LLVM_LIBC_FUNCTION(int, bcmp,
                   (const void *lhs, const void *rhs, size_t count)) {
  return inline_bcmp(lhs, rhs, count);
}

} // namespace LIBC_NAMESPACE_DECL
