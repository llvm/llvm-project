//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provide bit utilities for the flat_tlsf allocator.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_BIT_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_BIT_UTILS_H

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/flat_tlsf/common.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/math_extras.h"

namespace LIBC_NAMESPACE_DECL {
namespace flat_tlsf {

// We could use cpp::byte, but that definition currently have strict aliasing
// violation, hence we fallback to unsigned char directly.
namespace bit_utils {

LIBC_INLINE constexpr size_t ilog2(size_t n) {
  return cpp::numeric_limits<size_t>::digits - 1 -
         static_cast<size_t>(cpp::countl_zero(n));
}

LIBC_INLINE constexpr bool is_power_of_2(size_t n) {
  return cpp::has_single_bit(n);
}

LIBC_INLINE constexpr uint32_t bit_scan_after(size_t w, uint32_t start_index) {
  size_t lower_bits_cleared = (w >> start_index) << start_index;
  return static_cast<uint32_t>(cpp::countr_zero(lower_bits_cleared));
}

LIBC_INLINE constexpr void set_bit(size_t &w, uint32_t index) {
  w |= size_t{1} << index;
}

LIBC_INLINE constexpr void clear_bit(size_t &w, uint32_t index) {
  w &= ~(size_t{1} << index);
}

LIBC_INLINE constexpr bool read_bit(size_t w, uint32_t index) {
  return w & (size_t{1} << index);
}

LIBC_INLINE bool is_aligned_to(Byte *ptr, size_t align) {
  return (cpp::bit_cast<uintptr_t>(ptr) & (align - 1)) == 0;
}

LIBC_INLINE Byte *align_down_by(Byte *ptr, size_t align) {
  uintptr_t addr = cpp::bit_cast<uintptr_t>(ptr);
  return cpp::bit_cast<Byte *>(addr & ~(align - 1));
}

LIBC_INLINE Byte *align_up_by_mask(Byte *ptr, size_t align_mask) {
  uintptr_t addr = cpp::bit_cast<uintptr_t>(ptr);
  return cpp::bit_cast<Byte *>((addr + align_mask) & ~align_mask);
}

LIBC_INLINE Byte *align_up_by(Byte *ptr, size_t align) {
  return align_up_by_mask(ptr, align - 1);
}

LIBC_INLINE Byte *saturating_ptr_add(Byte *ptr, size_t bytes) {
  uintptr_t addr = cpp::bit_cast<uintptr_t>(ptr);
  uintptr_t result;
  if (add_overflow(addr, bytes, result))
    return cpp::bit_cast<Byte *>(cpp::numeric_limits<uintptr_t>::max());
  return cpp::bit_cast<Byte *>(result);
}

} // namespace bit_utils
} // namespace flat_tlsf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FLAT_TLSF_BIT_UTILS_H
