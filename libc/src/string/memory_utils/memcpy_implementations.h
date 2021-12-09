//===-- Memcpy implementation -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_IMPLEMENTATIONS_H

#include "src/__support/architectures.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/elements.h"
#include "src/string/memory_utils/utils.h"

#include <stddef.h> // size_t

// Design rationale
// ================
//
// Using a profiler to observe size distributions for calls into libc
// functions, it was found most operations act on a small number of bytes.
// This makes it important to favor small sizes.
//
// The tests for `count` are in ascending order so the cost of branching is
// proportional to the cost of copying.
//
// The function is written in C++ for several reasons:
// - The compiler can __see__ the code, this is useful when performing Profile
//   Guided Optimization as the optimized code can take advantage of branching
//   probabilities.
// - It also allows for easier customization and favors testing multiple
//   implementation parameters.
// - As compilers and processors get better, the generated code is improved
//   with little change on the code side.

namespace __llvm_libc {

static inline void inline_memcpy(char *__restrict dst,
                                 const char *__restrict src, size_t count) {
#if defined(LLVM_LIBC_ARCH_X86)
  /////////////////////////////////////////////////////////////////////////////
  // LLVM_LIBC_ARCH_X86
  /////////////////////////////////////////////////////////////////////////////
  using namespace __llvm_libc::x86;

  // Whether to use only rep;movsb.
  constexpr bool USE_ONLY_REP_MOVSB =
      LLVM_LIBC_IS_DEFINED(LLVM_LIBC_MEMCPY_X86_USE_ONLY_REPMOVSB);

  // kRepMovsBSize == -1 : Only CopyAligned is used.
  // kRepMovsBSize ==  0 : Only RepMovsb is used.
  // else CopyAligned is used up to kRepMovsBSize and then RepMovsb.
  constexpr size_t REP_MOVS_B_SIZE =
#if defined(LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE)
      LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE;
#else
      -1;
#endif // LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE

  // Whether target supports AVX instructions.
  constexpr bool HAS_AVX = LLVM_LIBC_IS_DEFINED(__AVX__);

#if defined(__AVX__)
  using LoopBlockSize = _64;
#else
  using LoopBlockSize = _32;
#endif

  if (USE_ONLY_REP_MOVSB)
    return copy<Accelerator>(dst, src, count);

  if (count == 0)
    return;
  if (count == 1)
    return copy<_1>(dst, src);
  if (count == 2)
    return copy<_2>(dst, src);
  if (count == 3)
    return copy<_3>(dst, src);
  if (count == 4)
    return copy<_4>(dst, src);
  if (count < 8)
    return copy<HeadTail<_4>>(dst, src, count);
  if (count < 16)
    return copy<HeadTail<_8>>(dst, src, count);
  if (count < 32)
    return copy<HeadTail<_16>>(dst, src, count);
  if (count < 64)
    return copy<HeadTail<_32>>(dst, src, count);
  if (count < 128)
    return copy<HeadTail<_64>>(dst, src, count);
  if (HAS_AVX && count < 256)
    return copy<HeadTail<_128>>(dst, src, count);
  if (count <= REP_MOVS_B_SIZE)
    return copy<Align<_32, Arg::Dst>::Then<Loop<LoopBlockSize>>>(dst, src,
                                                                 count);
  return copy<Accelerator>(dst, src, count);
#elif defined(LLVM_LIBC_ARCH_AARCH64)
  /////////////////////////////////////////////////////////////////////////////
  // LLVM_LIBC_ARCH_AARCH64
  /////////////////////////////////////////////////////////////////////////////
  using namespace __llvm_libc::scalar;
  if (count == 0)
    return;
  if (count == 1)
    return copy<_1>(dst, src);
  if (count == 2)
    return copy<_2>(dst, src);
  if (count == 3)
    return copy<_3>(dst, src);
  if (count == 4)
    return copy<_4>(dst, src);
  if (count < 8)
    return copy<HeadTail<_4>>(dst, src, count);
  if (count < 16)
    return copy<HeadTail<_8>>(dst, src, count);
  if (count < 32)
    return copy<HeadTail<_16>>(dst, src, count);
  if (count < 64)
    return copy<HeadTail<_32>>(dst, src, count);
  if (count < 128)
    return copy<HeadTail<_64>>(dst, src, count);
  return copy<Align<_16, Arg::Src>::Then<Loop<_64>>>(dst, src, count);
#else
  /////////////////////////////////////////////////////////////////////////////
  // Default
  /////////////////////////////////////////////////////////////////////////////
  using namespace __llvm_libc::scalar;
  if (count == 0)
    return;
  if (count == 1)
    return copy<_1>(dst, src);
  if (count == 2)
    return copy<_2>(dst, src);
  if (count == 3)
    return copy<_3>(dst, src);
  if (count == 4)
    return copy<_4>(dst, src);
  if (count < 8)
    return copy<HeadTail<_4>>(dst, src, count);
  if (count < 16)
    return copy<HeadTail<_8>>(dst, src, count);
  if (count < 32)
    return copy<HeadTail<_16>>(dst, src, count);
  if (count < 64)
    return copy<HeadTail<_32>>(dst, src, count);
  if (count < 128)
    return copy<HeadTail<_64>>(dst, src, count);
  return copy<Align<_32, Arg::Src>::Then<Loop<_32>>>(dst, src, count);
#endif
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_IMPLEMENTATIONS_H
