//===-- Memcpy utils --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_UTILS_H
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_UTILS_H

#include "src/__support/sanitizer.h"
#include "src/string/memory_utils/utils.h"
#include <stddef.h> // size_t

// __builtin_memcpy_inline guarantees to never call external functions.
// Unfortunately it is not widely available.
#ifdef __clang__
#if __has_builtin(__builtin_memcpy_inline)
#define USE_BUILTIN_MEMCPY_INLINE
#endif
#elif defined(__GNUC__)
#define USE_BUILTIN_MEMCPY
#endif

namespace __llvm_libc {

// This is useful for testing.
#if defined(LLVM_LIBC_MEMCPY_MONITOR)
extern "C" void LLVM_LIBC_MEMCPY_MONITOR(char *__restrict,
                                         const char *__restrict, size_t);
#endif

// Copies `kBlockSize` bytes from `src` to `dst` using a for loop.
// This code requires the use of `-fno-buitin-memcpy` to prevent the compiler
// from turning the for-loop back into `__builtin_memcpy`.
template <size_t kBlockSize>
static void ForLoopCopy(char *__restrict dst, const char *__restrict src) {
  for (size_t i = 0; i < kBlockSize; ++i)
    dst[i] = src[i];
}

// Copies `kBlockSize` bytes from `src` to `dst`.
template <size_t kBlockSize>
static void CopyBlock(char *__restrict dst, const char *__restrict src) {
#if defined(LLVM_LIBC_MEMCPY_MONITOR)
  LLVM_LIBC_MEMCPY_MONITOR(dst, src, kBlockSize);
#elif LLVM_LIBC_HAVE_MEMORY_SANITIZER || LLVM_LIBC_HAVE_ADDRESS_SANITIZER
  ForLoopCopy<kBlockSize>(dst, src);
#elif defined(USE_BUILTIN_MEMCPY_INLINE)
  __builtin_memcpy_inline(dst, src, kBlockSize);
#elif defined(USE_BUILTIN_MEMCPY)
  __builtin_memcpy(dst, src, kBlockSize);
#else
  ForLoopCopy<kBlockSize>(dst, src);
#endif
}

// Copies `kBlockSize` bytes from `src + count - kBlockSize` to
// `dst + count - kBlockSize`.
// Precondition: `count >= kBlockSize`.
template <size_t kBlockSize>
static void CopyLastBlock(char *__restrict dst, const char *__restrict src,
                          size_t count) {
  const size_t offset = count - kBlockSize;
  CopyBlock<kBlockSize>(dst + offset, src + offset);
}

// Copies `kBlockSize` bytes twice with an overlap between the two.
//
// [1234567812345678123]
// [__XXXXXXXXXXXXXX___]
// [__XXXXXXXX_________]
// [________XXXXXXXX___]
//
// Precondition: `count >= kBlockSize && count <= kBlockSize`.
template <size_t kBlockSize>
static void CopyBlockOverlap(char *__restrict dst, const char *__restrict src,
                             size_t count) {
  CopyBlock<kBlockSize>(dst, src);
  CopyLastBlock<kBlockSize>(dst, src, count);
}

// Copies `count` bytes by blocks of `kBlockSize` bytes.
// Copies at the start and end of the buffer are unaligned.
// Copies in the middle of the buffer are aligned to `kAlignment`.
//
// e.g. with
// [12345678123456781234567812345678]
// [__XXXXXXXXXXXXXXXXXXXXXXXXXXXX___]
// [__XXXX___________________________]
// [_____XXXXXXXX____________________]
// [_____________XXXXXXXX____________]
// [_____________________XXXXXXXX____]
// [______________________XXXXXXXX___]
//
// Precondition: `kAlignment <= kBlockSize`
//               `count > 2 * kBlockSize` for efficiency.
//               `count >= kAlignment` for correctness.
template <size_t kBlockSize, size_t kAlignment = kBlockSize>
static void CopyAlignedBlocks(char *__restrict dst, const char *__restrict src,
                              size_t count) {
  static_assert(is_power2(kAlignment), "kAlignment must be a power of two");
  static_assert(is_power2(kBlockSize), "kBlockSize must be a power of two");
  static_assert(kAlignment <= kBlockSize,
                "kAlignment must be less or equal to block size");
  CopyBlock<kAlignment>(dst, src); // Copy first block

  // Copy aligned blocks
  const size_t ofla = offset_from_last_aligned<kAlignment>(src);
  const size_t limit = count + ofla - kBlockSize;
  for (size_t offset = kAlignment; offset < limit; offset += kBlockSize)
    CopyBlock<kBlockSize>(dst - ofla + offset,
                          assume_aligned<kAlignment>(src - ofla + offset));

  CopyLastBlock<kBlockSize>(dst, src, count); // Copy last block
}

} // namespace __llvm_libc

#endif //  LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_UTILS_H
