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
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/op_x86.h"
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
  using namespace __llvm_libc::builtin;
#if defined(LLVM_LIBC_ARCH_X86)
  /////////////////////////////////////////////////////////////////////////////
  // LLVM_LIBC_ARCH_X86
  /////////////////////////////////////////////////////////////////////////////

  // Whether to use rep;movsb exclusively, not at all, or only above a certain
  // threshold.
  // TODO: Use only a single preprocessor definition to simplify the code.
#ifndef LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE
#define LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE -1
#endif

  static constexpr bool kUseOnlyRepMovsb =
      LLVM_LIBC_IS_DEFINED(LLVM_LIBC_MEMCPY_X86_USE_ONLY_REPMOVSB);
  static constexpr size_t kRepMovsbThreshold =
      LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE;

  if constexpr (kUseOnlyRepMovsb)
    return x86::Memcpy::repmovsb(dst, src, count);

  if (count == 0)
    return;
  if (count == 1)
    return Memcpy<1>::block(dst, src);
  if (count == 2)
    return Memcpy<2>::block(dst, src);
  if (count == 3)
    return Memcpy<3>::block(dst, src);
  if (count == 4)
    return Memcpy<4>::block(dst, src);
  if (count < 8)
    return Memcpy<4>::head_tail(dst, src, count);
  if (count < 16)
    return Memcpy<8>::head_tail(dst, src, count);
  if (count < 32)
    return Memcpy<16>::head_tail(dst, src, count);
  if (count < 64)
    return Memcpy<32>::head_tail(dst, src, count);
  if (count < 128)
    return Memcpy<64>::head_tail(dst, src, count);
  if (x86::kAvx && count < 256)
    return Memcpy<128>::head_tail(dst, src, count);
  if (count <= kRepMovsbThreshold) {
    Memcpy<32>::block(dst, src);
    align_to_next_boundary<32, Arg::Dst>(dst, src, count);
    return Memcpy < x86::kAvx ? 64 : 32 > ::loop_and_tail(dst, src, count);
  }
  return x86::Memcpy::repmovsb(dst, src, count);
#elif defined(LLVM_LIBC_ARCH_AARCH64)
  /////////////////////////////////////////////////////////////////////////////
  // LLVM_LIBC_ARCH_AARCH64
  /////////////////////////////////////////////////////////////////////////////
  if (count == 0)
    return;
  if (count == 1)
    return Memcpy<1>::block(dst, src);
  if (count == 2)
    return Memcpy<2>::block(dst, src);
  if (count == 3)
    return Memcpy<3>::block(dst, src);
  if (count == 4)
    return Memcpy<4>::block(dst, src);
  if (count < 8)
    return Memcpy<4>::head_tail(dst, src, count);
  if (count < 16)
    return Memcpy<8>::head_tail(dst, src, count);
  if (count < 32)
    return Memcpy<16>::head_tail(dst, src, count);
  if (count < 64)
    return Memcpy<32>::head_tail(dst, src, count);
  if (count < 128)
    return Memcpy<64>::head_tail(dst, src, count);
  Memcpy<16>::block(dst, src);
  align_to_next_boundary<16, Arg::Src>(dst, src, count);
  return Memcpy<64>::loop_and_tail(dst, src, count);
#else
  /////////////////////////////////////////////////////////////////////////////
  // Default
  /////////////////////////////////////////////////////////////////////////////
  if (count == 0)
    return;
  if (count == 1)
    return Memcpy<1>::block(dst, src);
  if (count == 2)
    return Memcpy<2>::block(dst, src);
  if (count == 3)
    return Memcpy<3>::block(dst, src);
  if (count == 4)
    return Memcpy<4>::block(dst, src);
  if (count < 8)
    return Memcpy<4>::head_tail(dst, src, count);
  if (count < 16)
    return Memcpy<8>::head_tail(dst, src, count);
  if (count < 32)
    return Memcpy<16>::head_tail(dst, src, count);
  if (count < 64)
    return Memcpy<32>::head_tail(dst, src, count);
  if (count < 128)
    return Memcpy<64>::head_tail(dst, src, count);
  Memcpy<32>::block(dst, src);
  align_to_next_boundary<32, Arg::Src>(dst, src, count);
  return Memcpy<32>::loop_and_tail(dst, src, count);
#endif
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_IMPLEMENTATIONS_H
