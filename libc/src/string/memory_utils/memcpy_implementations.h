//===-- Memcpy implementation -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_IMPLEMENTATIONS_H

#include "src/__support/common.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY LIBC_LOOP_NOUNROLL
#include "src/__support/macros/properties/architectures.h"
#include "src/string/memory_utils/op_aarch64.h"
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_x86.h"
#include "src/string/memory_utils/utils.h"

#include <stddef.h> // size_t

namespace __llvm_libc {

[[maybe_unused]] LIBC_INLINE void
inline_memcpy_embedded_tiny(Ptr __restrict dst, CPtr __restrict src,
                            size_t count) {
  LIBC_LOOP_NOUNROLL
  for (size_t offset = 0; offset < count; ++offset)
    builtin::Memcpy<1>::block(dst + offset, src + offset);
}

#if defined(LIBC_TARGET_ARCH_IS_X86)
[[maybe_unused]] LIBC_INLINE void
inline_memcpy_x86(Ptr __restrict dst, CPtr __restrict src, size_t count) {
  if (count == 0)
    return;
  if (count == 1)
    return builtin::Memcpy<1>::block(dst, src);
  if (count == 2)
    return builtin::Memcpy<2>::block(dst, src);
  if (count == 3)
    return builtin::Memcpy<3>::block(dst, src);
  if (count == 4)
    return builtin::Memcpy<4>::block(dst, src);
  if (count < 8)
    return builtin::Memcpy<4>::head_tail(dst, src, count);
  if (count < 16)
    return builtin::Memcpy<8>::head_tail(dst, src, count);
  if (count < 32)
    return builtin::Memcpy<16>::head_tail(dst, src, count);
  if (count < 64)
    return builtin::Memcpy<32>::head_tail(dst, src, count);
  if (count < 128)
    return builtin::Memcpy<64>::head_tail(dst, src, count);
  if (x86::kAvx && count < 256)
    return builtin::Memcpy<128>::head_tail(dst, src, count);
  builtin::Memcpy<32>::block(dst, src);
  align_to_next_boundary<32, Arg::Dst>(dst, src, count);
  static constexpr size_t kBlockSize = x86::kAvx ? 64 : 32;
  return builtin::Memcpy<kBlockSize>::loop_and_tail(dst, src, count);
}

[[maybe_unused]] LIBC_INLINE void
inline_memcpy_x86_maybe_interpose_repmovsb(Ptr __restrict dst,
                                           CPtr __restrict src, size_t count) {
  // Whether to use rep;movsb exclusively, not at all, or only above a certain
  // threshold.
#ifndef LIBC_COPT_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE
#define LIBC_COPT_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE -1
#endif

#ifdef LLVM_LIBC_MEMCPY_X86_USE_ONLY_REPMOVSB
#error LLVM_LIBC_MEMCPY_X86_USE_ONLY_REPMOVSB is deprecated use LIBC_COPT_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE=0 instead.
#endif // LLVM_LIBC_MEMCPY_X86_USE_ONLY_REPMOVSB

#ifdef LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE
#error LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE is deprecated use LIBC_COPT_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE=0 instead.
#endif // LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE

  static constexpr size_t kRepMovsbThreshold =
      LIBC_COPT_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE;
  if constexpr (kRepMovsbThreshold == 0) {
    return x86::Memcpy::repmovsb(dst, src, count);
  } else if constexpr (kRepMovsbThreshold == size_t(-1)) {
    return inline_memcpy_x86(dst, src, count);
  } else {
    if (LIBC_UNLIKELY(count >= kRepMovsbThreshold))
      return x86::Memcpy::repmovsb(dst, src, count);
    else
      return inline_memcpy_x86(dst, src, count);
  }
}
#endif // defined(LIBC_TARGET_ARCH_IS_X86)

#if defined(LIBC_TARGET_ARCH_IS_AARCH64)
[[maybe_unused]] LIBC_INLINE void
inline_memcpy_aarch64(Ptr __restrict dst, CPtr __restrict src, size_t count) {
  if (count == 0)
    return;
  if (count == 1)
    return builtin::Memcpy<1>::block(dst, src);
  if (count == 2)
    return builtin::Memcpy<2>::block(dst, src);
  if (count == 3)
    return builtin::Memcpy<3>::block(dst, src);
  if (count == 4)
    return builtin::Memcpy<4>::block(dst, src);
  if (count < 8)
    return builtin::Memcpy<4>::head_tail(dst, src, count);
  if (count < 16)
    return builtin::Memcpy<8>::head_tail(dst, src, count);
  if (count < 32)
    return builtin::Memcpy<16>::head_tail(dst, src, count);
  if (count < 64)
    return builtin::Memcpy<32>::head_tail(dst, src, count);
  if (count < 128)
    return builtin::Memcpy<64>::head_tail(dst, src, count);
  builtin::Memcpy<16>::block(dst, src);
  align_to_next_boundary<16, Arg::Src>(dst, src, count);
  return builtin::Memcpy<64>::loop_and_tail(dst, src, count);
}
#endif // defined(LIBC_TARGET_ARCH_IS_AARCH64)

LIBC_INLINE void inline_memcpy(Ptr __restrict dst, CPtr __restrict src,
                               size_t count) {
  using namespace __llvm_libc::builtin;
#if defined(LIBC_COPT_MEMCPY_USE_EMBEDDED_TINY) ||                             \
    defined(LIBC_TARGET_ARCH_IS_ARM) || defined(LIBC_TARGET_ARCH_IS_GPU)
  return inline_memcpy_embedded_tiny(dst, src, count);
#elif defined(LIBC_TARGET_ARCH_IS_X86)
  return inline_memcpy_x86_maybe_interpose_repmovsb(dst, src, count);
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  return inline_memcpy_aarch64(dst, src, count);
#else
#error "Unsupported platform"
#endif
}

LIBC_INLINE void inline_memcpy(void *__restrict dst, const void *__restrict src,
                               size_t count) {
  inline_memcpy(reinterpret_cast<Ptr>(dst), reinterpret_cast<CPtr>(src), count);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_IMPLEMENTATIONS_H
