//===-- Implementation of memset and bzero --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMSET_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMSET_IMPLEMENTATIONS_H

#include "src/__support/common.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/string/memory_utils/op_aarch64.h"
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_x86.h"
#include "src/string/memory_utils/utils.h"

#include <stddef.h> // size_t

namespace __llvm_libc {

[[maybe_unused]] LIBC_INLINE static void
inline_memset_embedded_tiny(Ptr dst, uint8_t value, size_t count) {
  LIBC_LOOP_NOUNROLL
  for (size_t offset = 0; offset < count; ++offset)
    generic::Memset<1, 1>::block(dst + offset, value);
}

#if defined(LIBC_TARGET_ARCH_IS_X86)
template <size_t MaxSize>
[[maybe_unused]] LIBC_INLINE static void
inline_memset_x86(Ptr dst, uint8_t value, size_t count) {
  if (count == 0)
    return;
  if (count == 1)
    return generic::Memset<1, MaxSize>::block(dst, value);
  if (count == 2)
    return generic::Memset<2, MaxSize>::block(dst, value);
  if (count == 3)
    return generic::Memset<3, MaxSize>::block(dst, value);
  if (count <= 8)
    return generic::Memset<4, MaxSize>::head_tail(dst, value, count);
  if (count <= 16)
    return generic::Memset<8, MaxSize>::head_tail(dst, value, count);
  if (count <= 32)
    return generic::Memset<16, MaxSize>::head_tail(dst, value, count);
  if (count <= 64)
    return generic::Memset<32, MaxSize>::head_tail(dst, value, count);
  if (count <= 128)
    return generic::Memset<64, MaxSize>::head_tail(dst, value, count);
  // Aligned loop
  generic::Memset<32, MaxSize>::block(dst, value);
  align_to_next_boundary<32>(dst, count);
  return generic::Memset<32, MaxSize>::loop_and_tail(dst, value, count);
}
#endif // defined(LIBC_TARGET_ARCH_IS_X86)

#if defined(LIBC_TARGET_ARCH_IS_AARCH64)
template <size_t MaxSize>
[[maybe_unused]] LIBC_INLINE static void
inline_memset_aarch64(Ptr dst, uint8_t value, size_t count) {
  if (count == 0)
    return;
  if (count <= 3) {
    generic::Memset<1, MaxSize>::block(dst, value);
    if (count > 1)
      generic::Memset<2, MaxSize>::tail(dst, value, count);
    return;
  }
  if (count <= 8)
    return generic::Memset<4, MaxSize>::head_tail(dst, value, count);
  if (count <= 16)
    return generic::Memset<8, MaxSize>::head_tail(dst, value, count);
  if (count <= 32)
    return generic::Memset<16, MaxSize>::head_tail(dst, value, count);
  if (count <= (32 + 64)) {
    generic::Memset<32, MaxSize>::block(dst, value);
    if (count <= 64)
      return generic::Memset<32, MaxSize>::tail(dst, value, count);
    generic::Memset<32, MaxSize>::block(dst + 32, value);
    generic::Memset<32, MaxSize>::tail(dst, value, count);
    return;
  }
  if (count >= 448 && value == 0 && aarch64::neon::hasZva()) {
    generic::Memset<64, MaxSize>::block(dst, 0);
    align_to_next_boundary<64>(dst, count);
    return aarch64::neon::BzeroCacheLine<64>::loop_and_tail(dst, 0, count);
  } else {
    generic::Memset<16, MaxSize>::block(dst, value);
    align_to_next_boundary<16>(dst, count);
    return generic::Memset<64, MaxSize>::loop_and_tail(dst, value, count);
  }
}
#endif // defined(LIBC_TARGET_ARCH_IS_AARCH64)

LIBC_INLINE static void inline_memset(Ptr dst, uint8_t value, size_t count) {
#if defined(LIBC_TARGET_ARCH_IS_X86)
  static constexpr size_t kMaxSize = x86::kAvx512F ? 64
                                     : x86::kAvx   ? 32
                                     : x86::kSse2  ? 16
                                                   : 8;
  return inline_memset_x86<kMaxSize>(dst, value, count);
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  static constexpr size_t kMaxSize = aarch64::kNeon ? 16 : 8;
  return inline_memset_aarch64<kMaxSize>(dst, value, count);
#elif defined(LIBC_TARGET_ARCH_IS_ARM)
  return inline_memset_embedded_tiny(dst, value, count);
#elif defined(LIBC_TARGET_ARCH_IS_GPU)
  return inline_memset_embedded_tiny(dst, value, count);
#else
#error "Unsupported platform"
#endif
}

LIBC_INLINE static void inline_memset(void *dst, uint8_t value, size_t count) {
  inline_memset(reinterpret_cast<Ptr>(dst), value, count);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMSET_IMPLEMENTATIONS_H
