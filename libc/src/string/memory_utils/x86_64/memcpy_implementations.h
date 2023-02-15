//===-- Memcpy implementation for x86_64 ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LIBC_SRC_STRING_MEMORY_UTILS_X86_64_MEMCPY_IMPLEMENTATIONS_H
#define LIBC_SRC_STRING_MEMORY_UTILS_X86_64_MEMCPY_IMPLEMENTATIONS_H

#include "src/__support/macros/config.h"       // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/op_x86.h"
#include "src/string/memory_utils/utils.h"

#include <stddef.h> // size_t

namespace __llvm_libc {

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

} // namespace __llvm_libc

#endif // LIBC_SRC_STRING_MEMORY_UTILS_X86_64_MEMCPY_IMPLEMENTATIONS_H
