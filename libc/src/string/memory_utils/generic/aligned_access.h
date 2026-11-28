//===-- Implementations for platform with mandatory aligned memory access -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// For some platforms, unaligned loads and stores are either illegal or very
// slow. The implementations in this file make sure all loads and stores are
// always aligned.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_ALIGNED_ACCESS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_ALIGNED_ACCESS_H

#include "src/__support/endian_internal.h"
#include "src/__support/macros/attributes.h" // LIBC_ALWAYS_INLINE, LIBC_INLINE
#include "src/__support/macros/config.h"     // LIBC_NAMESPACE_DECL
#include "src/__support/macros/optimization.h" // LIBC_LOOP_NOUNROLL
#include "src/string/memory_utils/generic/byte_per_byte.h"
#include "src/string/memory_utils/op_generic.h" // generic::splat
#include "src/string/memory_utils/utils.h"      // Ptr, CPtr

#include <stddef.h> // size_t

namespace LIBC_NAMESPACE_DECL {

[[maybe_unused]] LIBC_INLINE uint32_t load32_aligned(CPtr ptr, size_t offset,
                                                     size_t alignment) {
  if (alignment == 0)
    return load32_aligned<uint32_t>(ptr, offset);
  else if (alignment == 2)
    return load32_aligned<uint16_t, uint16_t>(ptr, offset);
  else // 1, 3
    return load32_aligned<uint8_t, uint16_t, uint8_t>(ptr, offset);
}

[[maybe_unused]] LIBC_INLINE uint64_t load64_aligned(CPtr ptr, size_t offset,
                                                     size_t alignment) {
  if (alignment == 0)
    return load64_aligned<uint64_t>(ptr, offset);
  else if (alignment == 4)
    return load64_aligned<uint32_t, uint32_t>(ptr, offset);
  else if (alignment == 6)
    return load64_aligned<uint16_t, uint32_t, uint16_t>(ptr, offset);
  else if (alignment == 2)
    return load64_aligned<uint16_t, uint16_t, uint16_t, uint16_t>(ptr, offset);
  else // 1, 3, 5, 7
    return load64_aligned<uint8_t, uint16_t, uint16_t, uint16_t, uint8_t>(
        ptr, offset);
}

///////////////////////////////////////////////////////////////////////////////
// memcpy
///////////////////////////////////////////////////////////////////////////////

[[maybe_unused]] LIBC_INLINE void
inline_memcpy_aligned_access_32bit(Ptr __restrict dst, CPtr __restrict src,
                                   size_t count) {
  constexpr size_t kAlign = sizeof(uint32_t);
  if (count <= 2 * kAlign)
    return inline_memcpy_byte_per_byte(dst, src, count);
  size_t bytes_to_dst_align = distance_to_align_up<kAlign>(dst);
  inline_memcpy_byte_per_byte(dst, src, bytes_to_dst_align);
  size_t offset = bytes_to_dst_align;
  size_t src_alignment = distance_to_align_down<kAlign>(src + offset);
  for (; offset < count - kAlign; offset += kAlign) {
    uint32_t value = load32_aligned(src, offset, src_alignment);
    store32_aligned<uint32_t>(value, dst, offset);
  }
  // remainder
  inline_memcpy_byte_per_byte(dst, src, count, offset);
}

[[maybe_unused]] LIBC_INLINE void
inline_memcpy_aligned_access_64bit(Ptr __restrict dst, CPtr __restrict src,
                                   size_t count) {
  constexpr size_t kAlign = sizeof(uint64_t);
  if (count <= 2 * kAlign)
    return inline_memcpy_byte_per_byte(dst, src, count);
  size_t bytes_to_dst_align = distance_to_align_up<kAlign>(dst);
  inline_memcpy_byte_per_byte(dst, src, bytes_to_dst_align);
  size_t offset = bytes_to_dst_align;
  size_t src_alignment = distance_to_align_down<kAlign>(src + offset);
  for (; offset < count - kAlign; offset += kAlign) {
    uint64_t value = load64_aligned(src, offset, src_alignment);
    store64_aligned<uint64_t>(value, dst, offset);
  }
  // remainder
  inline_memcpy_byte_per_byte(dst, src, count, offset);
}

///////////////////////////////////////////////////////////////////////////////
// memset
///////////////////////////////////////////////////////////////////////////////

[[maybe_unused]] LIBC_INLINE static void
inline_memset_aligned_access_32bit(Ptr dst, uint8_t value, size_t count) {
  constexpr size_t kAlign = sizeof(uint32_t);
  if (count <= 2 * kAlign)
    return inline_memset_byte_per_byte(dst, value, count);
  size_t bytes_to_dst_align = distance_to_align_up<kAlign>(dst);
  inline_memset_byte_per_byte(dst, value, bytes_to_dst_align);
  size_t offset = bytes_to_dst_align;
  for (; offset < count - kAlign; offset += kAlign)
    store32_aligned<uint32_t>(generic::splat<uint32_t>(value), dst, offset);
  inline_memset_byte_per_byte(dst, value, count, offset);
}

[[maybe_unused]] LIBC_INLINE static void
inline_memset_aligned_access_64bit(Ptr dst, uint8_t value, size_t count) {
  constexpr size_t kAlign = sizeof(uint64_t);
  if (count <= 2 * kAlign)
    return inline_memset_byte_per_byte(dst, value, count);
  size_t bytes_to_dst_align = distance_to_align_up<kAlign>(dst);
  inline_memset_byte_per_byte(dst, value, bytes_to_dst_align);
  size_t offset = bytes_to_dst_align;
  for (; offset < count - kAlign; offset += kAlign)
    store64_aligned<uint64_t>(generic::splat<uint64_t>(value), dst, offset);
  inline_memset_byte_per_byte(dst, value, count, offset);
}

///////////////////////////////////////////////////////////////////////////////
// bcmp
///////////////////////////////////////////////////////////////////////////////

template <typename Word>
LIBC_ALWAYS_INLINE BcmpReturnType inline_bcmp_aligned_access(CPtr p1, CPtr p2,
                                                             size_t count) {
  constexpr size_t ALIGN = sizeof(Word);
  if (count == 0)
    return BcmpReturnType::zero();
  if (count <= 3)
    return cpp::to_integer<uint32_t>((p1[0] ^ p2[0]) |
                                     (p1[count - 1] ^ p2[count - 1]) |
                                     (p1[count >> 1] ^ p2[count >> 1]));
  if (count >= 2 * ALIGN) {
    size_t lead = static_cast<size_t>(distance_to_align_up<ALIGN>(p1));
    const size_t p2_align =
        (static_cast<size_t>(distance_to_align_down<ALIGN>(p2)) + lead) &
        (ALIGN - 1);
    // When `p1` and `p2` are not mutually aligned, consume one extra leading
    // word byte-by-byte so `p1` stays aligned and `p2_base = p2 - p2_align`
    // lies strictly inside the caller's buffer (never reading outside
    // `[0, count)`).
    if (p2_align != 0)
      lead += ALIGN;
    if (lead != 0) {
      if (auto value = inline_bcmp_byte_per_byte(p1, p2, lead))
        return value;
      p1 += lead;
      p2 += lead;
      count -= lead;
    }
    auto load_word = [](CPtr p) -> Word {
      return load_unaligned<Word>(assume_aligned<ALIGN>(p));
    };
    if (p2_align == 0) {
      if constexpr (ALIGN == 8) {
        LIBC_LOOP_NOUNROLL
        for (; count >= 32; p1 += 32, p2 += 32, count -= 32) {
          if ((load_word(p1) ^ load_word(p2)) |
              (load_word(p1 + 8) ^ load_word(p2 + 8)) |
              (load_word(p1 + 16) ^ load_word(p2 + 16)) |
              (load_word(p1 + 24) ^ load_word(p2 + 24)))
            return BcmpReturnType::nonzero();
        }
      }
      LIBC_LOOP_NOUNROLL
      for (; count >= ALIGN; p1 += ALIGN, p2 += ALIGN, count -= ALIGN) {
        if (load_word(p1) != load_word(p2))
          return BcmpReturnType::nonzero();
      }
    } else if (count >= 2 * ALIGN) {
      CPtr p2_base = p2 - p2_align;
      const unsigned sh_lo = static_cast<unsigned>(p2_align * 8);
      const unsigned sh_hi = static_cast<unsigned>((ALIGN * 8) - sh_lo);
      Word prev = load_word(p2_base);
      LIBC_LOOP_NOUNROLL
      for (; count >= 2 * ALIGN;
           p1 += ALIGN, p2_base += ALIGN, count -= ALIGN) {
        const Word next = load_word(p2_base + ALIGN);
        const Word w2 = Endian::IS_LITTLE ? (prev >> sh_lo) | (next << sh_hi)
                                          : (prev << sh_lo) | (next >> sh_hi);
        if (load_word(p1) != w2)
          return BcmpReturnType::nonzero();
        prev = next;
      }
      p2 = p2_base + p2_align;
    }
  }
  return inline_bcmp_byte_per_byte(p1, p2, count);
}

[[maybe_unused]] LIBC_ALWAYS_INLINE BcmpReturnType
inline_bcmp_aligned_access_32bit(CPtr p1, CPtr p2, size_t count) {
  return inline_bcmp_aligned_access<uint32_t>(p1, p2, count);
}

[[maybe_unused]] LIBC_ALWAYS_INLINE BcmpReturnType
inline_bcmp_aligned_access_64bit(CPtr p1, CPtr p2, size_t count) {
  return inline_bcmp_aligned_access<uint64_t>(p1, p2, count);
}

///////////////////////////////////////////////////////////////////////////////
// memcmp
///////////////////////////////////////////////////////////////////////////////

[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_aligned_access_32bit(CPtr p1, CPtr p2, size_t count) {
  constexpr size_t kAlign = sizeof(uint32_t);
  if (count <= 2 * kAlign)
    return inline_memcmp_byte_per_byte(p1, p2, count);
  size_t bytes_to_p1_align = distance_to_align_up<kAlign>(p1);
  if (auto value = inline_memcmp_byte_per_byte(p1, p2, bytes_to_p1_align))
    return value;
  size_t offset = bytes_to_p1_align;
  size_t p2_alignment = distance_to_align_down<kAlign>(p2 + offset);
  for (; offset < count - kAlign; offset += kAlign) {
    uint32_t a = load32_aligned<uint32_t>(p1, offset);
    uint32_t b = load32_aligned(p2, offset, p2_alignment);
    if (a != b)
      return cmp_uint32_t(Endian::to_big_endian(a), Endian::to_big_endian(b));
  }
  return inline_memcmp_byte_per_byte(p1, p2, count, offset);
}

[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_aligned_access_64bit(CPtr p1, CPtr p2, size_t count) {
  constexpr size_t kAlign = sizeof(uint64_t);
  if (count <= 2 * kAlign)
    return inline_memcmp_byte_per_byte(p1, p2, count);
  size_t bytes_to_p1_align = distance_to_align_up<kAlign>(p1);
  if (auto value = inline_memcmp_byte_per_byte(p1, p2, bytes_to_p1_align))
    return value;
  size_t offset = bytes_to_p1_align;
  size_t p2_alignment = distance_to_align_down<kAlign>(p2 + offset);
  for (; offset < count - kAlign; offset += kAlign) {
    uint64_t a = load64_aligned<uint64_t>(p1, offset);
    uint64_t b = load64_aligned(p2, offset, p2_alignment);
    if (a != b)
      return cmp_neq_uint64_t(Endian::to_big_endian(a),
                              Endian::to_big_endian(b));
  }
  return inline_memcmp_byte_per_byte(p1, p2, count, offset);
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_GENERIC_ALIGNED_ACCESS_H
