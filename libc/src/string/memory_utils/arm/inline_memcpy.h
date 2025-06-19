#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ARM_INLINE_MEMCPY_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ARM_INLINE_MEMCPY_H

#include "src/__support/macros/attributes.h"   // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_LOOP_NOUNROLL
#include "src/string/memory_utils/utils.h" // memcpy_inline, distance_to_align

#include <stddef.h> // size_t

namespace LIBC_NAMESPACE_DECL {

namespace {

LIBC_INLINE_VAR constexpr size_t kWordSize = sizeof(uint32_t);

template <size_t bytes>
LIBC_INLINE void copy_and_bump_pointers(Ptr &dst, CPtr &src) {
  if constexpr (bytes == 1 || bytes == 2 || bytes == 4) {
    memcpy_inline<bytes>(dst, src);
  } else {
    // We restrict loads/stores to 4 byte to prevent the use of load/store
    // multiple (LDM, STM) and load/store double (LDRD, STRD). First, they may
    // fault (see notes below) and second, they use more registers which in turn
    // adds push/pop instructions in the hot path.
    static_assert(bytes % kWordSize == 0);
    LIBC_LOOP_UNROLL
    for (size_t i = 0; i < bytes / kWordSize; ++i) {
      const uintptr_t offset = i * kWordSize;
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

template <size_t block_size>
LIBC_INLINE void copy_blocks(Ptr &dst, CPtr &src, size_t &size) {
  LIBC_LOOP_NOUNROLL
  for (size_t i = 0; i < size / block_size; ++i)
    copy_and_bump_pointers<block_size>(dst, src);
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
// The implementation makes sure that all accesses are aligned.
[[maybe_unused]] LIBC_INLINE void inline_memcpy_arm_low_end(Ptr dst, CPtr src,
                                                            size_t size) {
  // For now, dummy implementation that performs byte per byte copy.
  LIBC_LOOP_NOUNROLL
  for (size_t i = 0; i < size; ++i)
    dst[i] = src[i];
}

// Implementation for Cortex-M3, M4, M7, M23, M33, M35P, M52 with hardware
// support for unaligned loads and stores.
// Notes:
// - It compiles down to <300 bytes.
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
  if (misaligned(bitwise_or(src, dst))) [[unlikely]] {
    if (size < 8) [[unlikely]] {
      if (size & 1)
        copy_and_bump_pointers<1>(dst, src);
      if (size & 2)
        copy_and_bump_pointers<2>(dst, src);
      if (size & 4)
        copy_and_bump_pointers<4>(dst, src);
      return;
    }
    if (misaligned(src)) [[unlikely]] {
      const size_t offset = distance_to_align_up<kWordSize>(dst);
      if (offset & 1)
        copy_and_bump_pointers<1>(dst, src);
      if (offset & 2)
        copy_and_bump_pointers<2>(dst, src);
      size -= offset;
    }
  }
  copy_blocks<64>(dst, src, size);
  copy_blocks<16>(dst, src, size);
  copy_blocks<4>(dst, src, size);
  if (size & 1)
    copy_and_bump_pointers<1>(dst, src);
  if (size & 2) [[unlikely]]
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

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ARM_INLINE_MEMCPY_H
