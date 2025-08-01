//===-- Memcpy implementation for arm ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ARM_INLINE_MEMCPY_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ARM_INLINE_MEMCPY_H

#include "src/__support/macros/attributes.h"   // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_LOOP_NOUNROLL
#include "src/string/memory_utils/utils.h" // memcpy_inline, distance_to_align

#include <stddef.h> // size_t

// https://libc.llvm.org/compiler_support.html
// Support for [[likely]] / [[unlikely]]
//  [X] GCC 12.2
//  [X] Clang 12
//  [ ] Clang 11
#define LIBC_ATTR_LIKELY [[likely]]
#define LIBC_ATTR_UNLIKELY [[unlikely]]

#if defined(LIBC_COMPILER_IS_CLANG)
#if LIBC_COMPILER_CLANG_VER < 1200
#undef LIBC_ATTR_LIKELY
#undef LIBC_ATTR_UNLIKELY
#define LIBC_ATTR_LIKELY
#define LIBC_ATTR_UNLIKELY
#endif
#endif

namespace LIBC_NAMESPACE_DECL {

namespace {

LIBC_INLINE_VAR constexpr size_t kWordSize = sizeof(uint32_t);

enum Strategy {
  ForceWordLdStChain,
  AssumeWordAligned,
  AssumeUnaligned,
};

template <size_t bytes, Strategy strategy = AssumeUnaligned>
LIBC_INLINE void copy_and_bump_pointers(Ptr &dst, CPtr &src) {
  if constexpr (strategy == AssumeUnaligned) {
    memcpy_inline<bytes>(assume_aligned<1>(dst), assume_aligned<1>(src));
  } else if constexpr (strategy == AssumeWordAligned) {
    static_assert(bytes >= kWordSize);
    memcpy_inline<bytes>(assume_aligned<kWordSize>(dst),
                         assume_aligned<kWordSize>(src));
  } else if constexpr (strategy == ForceWordLdStChain) {
    // We restrict loads/stores to 4 byte to prevent the use of load/store
    // multiple (LDM, STM) and load/store double (LDRD, STRD). First, they may
    // fault (see notes below) and second, they use more registers which in turn
    // adds push/pop instructions in the hot path.
    static_assert((bytes % kWordSize == 0) && (bytes >= kWordSize));
    LIBC_LOOP_UNROLL
    for (size_t i = 0; i < bytes / kWordSize; ++i) {
      const size_t offset = i * kWordSize;
      memcpy_inline<kWordSize>(dst + offset, src + offset);
    }
  }
  // In the 1, 2, 4 byte copy case, the compiler can fold pointer offsetting
  // into the load/store instructions.
  // e.g.,
  // ldrb  r3, [r1], #1
  // strb  r3, [r0], #1
  dst += bytes;
  src += bytes;
}

LIBC_INLINE void copy_bytes_and_bump_pointers(Ptr &dst, CPtr &src,
                                              const size_t size) {
  LIBC_LOOP_NOUNROLL
  for (size_t i = 0; i < size; ++i)
    *dst++ = *src++;
}

template <size_t block_size, Strategy strategy>
LIBC_INLINE void copy_blocks_and_update_args(Ptr &dst, CPtr &src,
                                             size_t &size) {
  LIBC_LOOP_NOUNROLL
  for (size_t i = 0; i < size / block_size; ++i)
    copy_and_bump_pointers<block_size, strategy>(dst, src);
  // Update `size` once at the end instead of once per iteration.
  size %= block_size;
}

LIBC_INLINE CPtr bitwise_or(CPtr a, CPtr b) {
  return cpp::bit_cast<CPtr>(cpp::bit_cast<uintptr_t>(a) |
                             cpp::bit_cast<uintptr_t>(b));
}

LIBC_INLINE auto misaligned(CPtr a) {
  return distance_to_align_down<kWordSize>(a);
}

} // namespace

// Implementation for Cortex-M0, M0+, M1.
// Notes:
// - It compiles down to 196 bytes, but 220 bytes when used through `memcpy`
//   that also needs to return the `dst` ptr.
// - These cores do not allow for unaligned loads/stores.
// - When `src` and `dst` are coaligned, we start by aligning them and perform
//   bulk copies. We let the compiler know the pointers are aligned so it can
//   use load/store multiple (LDM, STM). This significantly increase throughput
//   but it also requires more registers and push/pop instructions. This impacts
//   latency for small size copies.
// - When `src` and `dst` are misaligned, we align `dst` and recompose words
//   using multiple aligned loads. `load_aligned` takes care of endianness
//   issues.
[[maybe_unused]] LIBC_INLINE void inline_memcpy_arm_low_end(Ptr dst, CPtr src,
                                                            size_t size) {
  if (size >= 8) {
    if (const size_t offset = distance_to_align_up<kWordSize>(dst))
      LIBC_ATTR_UNLIKELY {
        copy_bytes_and_bump_pointers(dst, src, offset);
        size -= offset;
      }
    const auto src_alignment = distance_to_align_down<kWordSize>(src);
    if (src_alignment == 0)
      LIBC_ATTR_LIKELY {
        // Both `src` and `dst` are now word-aligned.
        copy_blocks_and_update_args<64, AssumeWordAligned>(dst, src, size);
        copy_blocks_and_update_args<16, AssumeWordAligned>(dst, src, size);
        copy_blocks_and_update_args<4, AssumeWordAligned>(dst, src, size);
      }
    else {
      // `dst` is aligned but `src` is not.
      LIBC_LOOP_NOUNROLL
      while (size >= kWordSize) {
        // Recompose word from multiple loads depending on the alignment.
        const uint32_t value =
            src_alignment == 2
                ? load_aligned<uint32_t, uint16_t, uint16_t>(src)
                : load_aligned<uint32_t, uint8_t, uint16_t, uint8_t>(src);
        memcpy_inline<kWordSize>(assume_aligned<kWordSize>(dst), &value);
        dst += kWordSize;
        src += kWordSize;
        size -= kWordSize;
      }
    }
    // Up to 3 bytes may still need to be copied.
    // Handling them with the slow loop below.
  }
  copy_bytes_and_bump_pointers(dst, src, size);
}

// Implementation for Cortex-M3, M4, M7, M23, M33, M35P, M52 with hardware
// support for unaligned loads and stores.
// Notes:
// - It compiles down to 266 bytes.
// - `dst` and `src` are not `__restrict` to prevent the compiler from
//   reordering loads/stores.
// - We keep state variables to a strict minimum to keep everything in the free
//   registers and prevent costly push / pop.
// - If unaligned single loads/stores to normal memory are supported, unaligned
//   accesses for load/store multiple (LDM, STM) and load/store double (LDRD,
//   STRD) instructions are generally not supported and will still fault so we
//   make sure to restrict unrolling to word loads/stores.
[[maybe_unused]] LIBC_INLINE void inline_memcpy_arm_mid_end(Ptr dst, CPtr src,
                                                            size_t size) {
  if (misaligned(bitwise_or(src, dst)))
    LIBC_ATTR_UNLIKELY {
      if (size < 8)
        LIBC_ATTR_UNLIKELY {
          if (size & 1)
            copy_and_bump_pointers<1>(dst, src);
          if (size & 2)
            copy_and_bump_pointers<2>(dst, src);
          if (size & 4)
            copy_and_bump_pointers<4>(dst, src);
          return;
        }
      if (misaligned(src))
        LIBC_ATTR_UNLIKELY {
          const size_t offset = distance_to_align_up<kWordSize>(dst);
          if (offset & 1)
            copy_and_bump_pointers<1>(dst, src);
          if (offset & 2)
            copy_and_bump_pointers<2>(dst, src);
          size -= offset;
        }
    }
  copy_blocks_and_update_args<64, ForceWordLdStChain>(dst, src, size);
  copy_blocks_and_update_args<16, ForceWordLdStChain>(dst, src, size);
  copy_blocks_and_update_args<4, AssumeUnaligned>(dst, src, size);
  if (size & 1)
    copy_and_bump_pointers<1>(dst, src);
  if (size & 2)
    LIBC_ATTR_UNLIKELY
  copy_and_bump_pointers<2>(dst, src);
}

[[maybe_unused]] LIBC_INLINE void inline_memcpy_arm(void *__restrict dst_,
                                                    const void *__restrict src_,
                                                    size_t size) {
  Ptr dst = cpp::bit_cast<Ptr>(dst_);
  CPtr src = cpp::bit_cast<CPtr>(src_);
#ifdef __ARM_FEATURE_UNALIGNED
  return inline_memcpy_arm_mid_end(dst, src, size);
#else
  return inline_memcpy_arm_low_end(dst, src, size);
#endif
}

} // namespace LIBC_NAMESPACE_DECL

// Cleanup local macros
#undef LIBC_ATTR_LIKELY
#undef LIBC_ATTR_UNLIKELY

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ARM_INLINE_MEMCPY_H
