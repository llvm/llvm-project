//===-- Memcpy utils --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_UTILS_H
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_UTILS_H

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

// Copies `kBlockSize` bytes from `src` to `dst`.
template <size_t kBlockSize>
static void Copy(char *__restrict dst, const char *__restrict src) {
#if defined(LLVM_LIBC_MEMCPY_MONITOR)
  LLVM_LIBC_MEMCPY_MONITOR(dst, src, kBlockSize);
#elif defined(USE_BUILTIN_MEMCPY_INLINE)
  __builtin_memcpy_inline(dst, src, kBlockSize);
#elif defined(USE_BUILTIN_MEMCPY)
  __builtin_memcpy(dst, src, kBlockSize);
#else
  for (size_t i = 0; i < kBlockSize; ++i)
    dst[i] = src[i];
#endif
}

// Copies `kBlockSize` bytes from `src + count - kBlockSize` to
// `dst + count - kBlockSize`.
// Precondition: `count >= kBlockSize`.
template <size_t kBlockSize>
static void CopyLastBlock(char *__restrict dst, const char *__restrict src,
                          size_t count) {
  const size_t offset = count - kBlockSize;
  Copy<kBlockSize>(dst + offset, src + offset);
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
static void CopyOverlap(char *__restrict dst, const char *__restrict src,
                        size_t count) {
  Copy<kBlockSize>(dst, src);
  CopyLastBlock<kBlockSize>(dst, src, count);
}

// Copies `count` bytes by blocks of `kBlockSize` bytes.
// Copies at the start and end of the buffer are unaligned.
// Copies in the middle of the buffer are aligned to `kBlockSize`.
//
// e.g. with
// [12345678123456781234567812345678]
// [__XXXXXXXXXXXXXXXXXXXXXXXXXXX___]
// [__XXXXXXXX______________________]
// [________XXXXXXXX________________]
// [________________XXXXXXXX________]
// [_____________________XXXXXXXX___]
//
// Precondition: `count > 2 * kBlockSize` for efficiency.
//               `count >= kBlockSize` for correctness.
template <size_t kBlockSize>
static void CopyAligned(char *__restrict dst, const char *__restrict src,
                        size_t count) {
  Copy<kBlockSize>(dst, src); // Copy first block

  // Copy aligned blocks
  size_t offset = kBlockSize - offset_from_last_aligned<kBlockSize>(dst);
  for (; offset + kBlockSize < count; offset += kBlockSize)
    Copy<kBlockSize>(dst + offset, src + offset);

  CopyLastBlock<kBlockSize>(dst, src, count); // Copy last block
}

} // namespace __llvm_libc

#endif //  LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_UTILS_H
