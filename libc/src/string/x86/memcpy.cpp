//===-- Implementation of memcpy ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memcpy.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/memcpy_utils.h"

namespace __llvm_libc {

static void CopyRepMovsb(char *__restrict dst, const char *__restrict src,
                         size_t count) {
  // FIXME: Add MSVC support with
  // #include <intrin.h>
  // __movsb(reinterpret_cast<unsigned char *>(dst),
  //         reinterpret_cast<const unsigned char *>(src), count);
  asm volatile("rep movsb" : "+D"(dst), "+S"(src), "+c"(count) : : "memory");
}

#if defined(__AVX__)
#define BEST_SIZE 64
#else
#define BEST_SIZE 32
#endif

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
static void memcpy_x86(char *__restrict dst, const char *__restrict src,
                       size_t count) {
  if (count == 0)
    return;
  if (count == 1)
    return CopyBlock<1>(dst, src);
  if (count == 2)
    return CopyBlock<2>(dst, src);
  if (count == 3)
    return CopyBlock<3>(dst, src);
  if (count == 4)
    return CopyBlock<4>(dst, src);
  if (count < 8)
    return CopyBlockOverlap<4>(dst, src, count);
  if (count < 16)
    return CopyBlockOverlap<8>(dst, src, count);
  if (count < 32)
    return CopyBlockOverlap<16>(dst, src, count);
  if (count < 64)
    return CopyBlockOverlap<32>(dst, src, count);
  if (count < 128)
    return CopyBlockOverlap<64>(dst, src, count);
#if defined(__AVX__)
  if (count < 256)
    return CopyBlockOverlap<128>(dst, src, count);
#endif
  // kRepMovsBSize == -1 : Only CopyAligned is used.
  // kRepMovsBSize ==  0 : Only RepMovsb is used.
  // else CopyAligned is used to to kRepMovsBSize and then RepMovsb.
  constexpr size_t kRepMovsBSize = -1;
  if (count <= kRepMovsBSize)
    return CopyAlignedBlocks<BEST_SIZE>(dst, src, count);
  return CopyRepMovsb(dst, src, count);
}

void *LLVM_LIBC_ENTRYPOINT(memcpy)(void *__restrict dst,
                                   const void *__restrict src, size_t size) {
  memcpy_x86(reinterpret_cast<char *>(dst), reinterpret_cast<const char *>(src),
             size);
  return dst;
}

} // namespace __llvm_libc
