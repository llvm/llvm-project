//===-- Memcpy implementation for x86_64 ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_MEMCPY_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_MEMCPY_H

#include "src/__support/macros/attributes.h"   // LIBC_INLINE_VAR
#include "src/__support/macros/config.h"       // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/op_x86.h"
#include "src/string/memory_utils/utils.h"

#include <stddef.h> // size_t
#include <stdint.h> // SIZE_MAX

#ifdef LLVM_LIBC_MEMCPY_X86_USE_ONLY_REPMOVSB
#error LLVM_LIBC_MEMCPY_X86_USE_ONLY_REPMOVSB is deprecated use LIBC_COPT_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE=0 instead.
#endif // LLVM_LIBC_MEMCPY_X86_USE_ONLY_REPMOVSB

#ifdef LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE
#error LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE is deprecated use LIBC_COPT_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE=0 instead.
#endif // LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE

namespace LIBC_NAMESPACE {

namespace x86 {

LIBC_INLINE_VAR constexpr size_t kOneCacheline = 64;
LIBC_INLINE_VAR constexpr size_t kTwoCachelines = 2 * kOneCacheline;
LIBC_INLINE_VAR constexpr size_t kThreeCachelines = 3 * kOneCacheline;

LIBC_INLINE_VAR constexpr bool kUseSoftwarePrefetching =
    LLVM_LIBC_IS_DEFINED(LIBC_COPT_MEMCPY_X86_USE_SOFTWARE_PREFETCHING);

// Whether to use rep;movsb exclusively (0), not at all (SIZE_MAX), or only
// above a certain threshold. Defaults to "do not use rep;movsb".
#ifndef LIBC_COPT_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE
#define LIBC_COPT_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE SIZE_MAX
#endif
LIBC_INLINE_VAR constexpr size_t kRepMovsbThreshold =
    LIBC_COPT_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE;

} // namespace x86

[[maybe_unused]] LIBC_INLINE void
inline_memcpy_x86_sse2_ge64(Ptr __restrict dst, CPtr __restrict src,
                            size_t count) {
  if (count <= 128)
    return builtin::Memcpy<64>::head_tail(dst, src, count);
  builtin::Memcpy<32>::block(dst, src);
  align_to_next_boundary<32, Arg::Dst>(dst, src, count);
  return builtin::Memcpy<32>::loop_and_tail(dst, src, count);
}

[[maybe_unused]] LIBC_INLINE void
inline_memcpy_x86_avx_ge64(Ptr __restrict dst, CPtr __restrict src,
                           size_t count) {
  if (count <= 128)
    return builtin::Memcpy<64>::head_tail(dst, src, count);
  if (count < 256)
    return builtin::Memcpy<128>::head_tail(dst, src, count);
  builtin::Memcpy<32>::block(dst, src);
  align_to_next_boundary<32, Arg::Dst>(dst, src, count);
  return builtin::Memcpy<64>::loop_and_tail(dst, src, count);
}

[[maybe_unused]] LIBC_INLINE void
inline_memcpy_x86_sse2_ge64_sw_prefetching(Ptr __restrict dst,
                                           CPtr __restrict src, size_t count) {
  using namespace LIBC_NAMESPACE::x86;
  prefetch_to_local_cache(src + kOneCacheline);
  if (count <= 128)
    return builtin::Memcpy<64>::head_tail(dst, src, count);
  prefetch_to_local_cache(src + kTwoCachelines);
  // Aligning 'dst' on a 32B boundary.
  builtin::Memcpy<32>::block(dst, src);
  align_to_next_boundary<32, Arg::Dst>(dst, src, count);
  builtin::Memcpy<96>::block(dst, src);
  size_t offset = 96;
  // At this point:
  // - we copied between 96B and 128B,
  // - we prefetched cachelines at 'src + 64' and 'src + 128',
  // - 'dst' is 32B aligned,
  // - count >= 128.
  if (count < 352) {
    // Two cache lines at a time.
    while (offset + kTwoCachelines + 32 <= count) {
      prefetch_to_local_cache(src + offset + kOneCacheline);
      prefetch_to_local_cache(src + offset + kTwoCachelines);
      builtin::Memcpy<kTwoCachelines>::block_offset(dst, src, offset);
      offset += kTwoCachelines;
    }
  } else {
    // Three cache lines at a time.
    while (offset + kThreeCachelines + 32 <= count) {
      prefetch_to_local_cache(src + offset + kOneCacheline);
      prefetch_to_local_cache(src + offset + kTwoCachelines);
      prefetch_to_local_cache(src + offset + kThreeCachelines);
      // It is likely that this copy will be turned into a 'rep;movsb' on
      // non-AVX machines.
      builtin::Memcpy<kThreeCachelines>::block_offset(dst, src, offset);
      offset += kThreeCachelines;
    }
  }
  return builtin::Memcpy<32>::loop_and_tail_offset(dst, src, count, offset);
}

[[maybe_unused]] LIBC_INLINE void
inline_memcpy_x86_avx_ge64_sw_prefetching(Ptr __restrict dst,
                                          CPtr __restrict src, size_t count) {
  using namespace LIBC_NAMESPACE::x86;
  prefetch_to_local_cache(src + kOneCacheline);
  if (count <= 128)
    return builtin::Memcpy<64>::head_tail(dst, src, count);
  prefetch_to_local_cache(src + kTwoCachelines);
  prefetch_to_local_cache(src + kThreeCachelines);
  if (count < 256)
    return builtin::Memcpy<128>::head_tail(dst, src, count);
  // Aligning 'dst' on a 32B boundary.
  builtin::Memcpy<32>::block(dst, src);
  align_to_next_boundary<32, Arg::Dst>(dst, src, count);
  builtin::Memcpy<224>::block(dst, src);
  size_t offset = 224;
  // At this point:
  // - we copied between 224B and 256B,
  // - we prefetched cachelines at 'src + 64', 'src + 128', and 'src + 196'
  // - 'dst' is 32B aligned,
  // - count >= 128.
  while (offset + kThreeCachelines + 64 <= count) {
    // Three cache lines at a time.
    prefetch_to_local_cache(src + offset + kOneCacheline);
    prefetch_to_local_cache(src + offset + kTwoCachelines);
    prefetch_to_local_cache(src + offset + kThreeCachelines);
    builtin::Memcpy<kThreeCachelines>::block_offset(dst, src, offset);
    offset += kThreeCachelines;
  }
  return builtin::Memcpy<64>::loop_and_tail_offset(dst, src, count, offset);
}

[[maybe_unused]] LIBC_INLINE void
inline_memcpy_x86(Ptr __restrict dst, CPtr __restrict src, size_t count) {
#if defined(__AVX512F__)
  constexpr size_t vector_size = 64;
#elif defined(__AVX__)
  constexpr size_t vector_size = 32;
#elif defined(__SSE2__)
  constexpr size_t vector_size = 16;
#else
  constexpr size_t vector_size = 8;
#endif
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
  // If count is equal to a power of 2, we can handle it as head-tail
  // of both smaller size and larger size (head-tail are either
  // non-overlapping for smaller size, or completely collapsed
  // for larger size). It seems to be more profitable to do the copy
  // with the larger size, if it's natively supported (e.g. doing
  // 2 collapsed 32-byte moves for count=64 if AVX2 is supported).
  // But it's not profitable to use larger size if it's not natively
  // supported: we will both use more instructions and handle fewer
  // sizes in earlier branches.
  if (vector_size >= 16 ? count < 16 : count <= 16)
    return builtin::Memcpy<8>::head_tail(dst, src, count);
  if (vector_size >= 32 ? count < 32 : count <= 32)
    return builtin::Memcpy<16>::head_tail(dst, src, count);
  if (vector_size >= 64 ? count < 64 : count <= 64)
    return builtin::Memcpy<32>::head_tail(dst, src, count);
  if constexpr (x86::kAvx) {
    if constexpr (x86::kUseSoftwarePrefetching) {
      return inline_memcpy_x86_avx_ge64_sw_prefetching(dst, src, count);
    } else {
      return inline_memcpy_x86_avx_ge64(dst, src, count);
    }
  } else {
    if constexpr (x86::kUseSoftwarePrefetching) {
      return inline_memcpy_x86_sse2_ge64_sw_prefetching(dst, src, count);
    } else {
      return inline_memcpy_x86_sse2_ge64(dst, src, count);
    }
  }
}

[[maybe_unused]] LIBC_INLINE void
inline_memcpy_x86_maybe_interpose_repmovsb(Ptr __restrict dst,
                                           CPtr __restrict src, size_t count) {
  if constexpr (x86::kRepMovsbThreshold == 0) {
    return x86::Memcpy::repmovsb(dst, src, count);
  } else if constexpr (x86::kRepMovsbThreshold == SIZE_MAX) {
    return inline_memcpy_x86(dst, src, count);
  } else {
    if (LIBC_UNLIKELY(count >= x86::kRepMovsbThreshold))
      return x86::Memcpy::repmovsb(dst, src, count);
    else
      return inline_memcpy_x86(dst, src, count);
  }
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_MEMCPY_H
