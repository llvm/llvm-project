//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the memcmp implementations for arm.
///
///===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ARM_INLINE_MEMCMP_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ARM_INLINE_MEMCMP_H

#include "src/__support/CPP/bit.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/cpu_features.h"
#include "src/__support/math_extras.h"
#include "src/string/memory_utils/generic/byte_per_byte.h"
#include "src/string/memory_utils/utils.h"

#if defined(LIBC_TARGET_CPU_HAS_MVE)
#include "src/__support/CPP/simd.h"
#endif

namespace LIBC_NAMESPACE_DECL {
// We limit code size and don't aggressively expand the loop to multiway
// parallel comparison.
[[maybe_unused]] LIBC_INLINE MemcmpReturnType inline_memcmp_arm(CPtr p1,
                                                                CPtr p2,
                                                                size_t count) {
#if defined(LIBC_TARGET_CPU_HAS_MVE)
  using Bytes = cpp::simd<uint8_t, 16>;
  using Mask = cpp::simd<bool, 16>;
  cpp::simd<uint32_t, 16> lanes = cpp::iota<uint32_t, 16>();
  // Cast to raw address to avoid ub:expr.add.out.of.bounds
  uintptr_t p1_addr = cpp::bit_cast<uintptr_t>(p1);
  uintptr_t p2_addr = cpp::bit_cast<uintptr_t>(p2);
  while (count != 0) {
    // Predication handles the final partial vector without reading past count.
    Mask active = cpp::simd_cast<bool>(lanes < count);
    Bytes a =
        cpp::load_masked<Bytes>(active, cpp::bit_cast<CPtr>(p1_addr), Bytes{});
    Bytes b =
        cpp::load_masked<Bytes>(active, cpp::bit_cast<CPtr>(p2_addr), Bytes{});
    Mask mismatches = cpp::simd_cast<bool>(a != b) & active;
    if (cpp::any_of(mismatches)) {
      size_t offset = cpp::find_first_set(mismatches);
      return static_cast<int32_t>(cpp::bit_cast<CPtr>(p1_addr)[offset]) -
             static_cast<int32_t>(cpp::bit_cast<CPtr>(p2_addr)[offset]);
    }
    // optimistically increase the address
    p1_addr += 16;
    p2_addr += 16;
    if (sub_overflow(count, size_t{16}, count))
      break;
  }
  return MemcmpReturnType::zero();
#else
  return inline_memcmp_byte_per_byte(p1, p2, count);
#endif
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ARM_INLINE_MEMCMP_H
