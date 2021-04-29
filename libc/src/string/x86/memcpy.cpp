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

// Whether to use only rep;movsb.
constexpr bool kUseOnlyRepMovsb =
    LLVM_LIBC_IS_DEFINED(LLVM_LIBC_MEMCPY_X86_USE_ONLY_REPMOVSB);

// kRepMovsBSize == -1 : Only CopyAligned is used.
// kRepMovsBSize ==  0 : Only RepMovsb is used.
// else CopyAligned is used up to kRepMovsBSize and then RepMovsb.
constexpr size_t kRepMovsBSize =
#ifdef LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE
    LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE;
#else
    -1;
#endif // LLVM_LIBC_MEMCPY_X86_USE_REPMOVSB_FROM_SIZE

// Whether target supports AVX instructions.
constexpr bool kHasAvx = LLVM_LIBC_IS_DEFINED(__AVX__);

// The chunk size used for the loop copy strategy.
constexpr size_t kLoopCopyBlockSize = kHasAvx ? 64 : 32;

static void CopyRepMovsb(char *__restrict dst, const char *__restrict src,
                         size_t count) {
  // FIXME: Add MSVC support with
  // #include <intrin.h>
  // __movsb(reinterpret_cast<unsigned char *>(dst),
  //         reinterpret_cast<const unsigned char *>(src), count);
  asm volatile("rep movsb" : "+D"(dst), "+S"(src), "+c"(count) : : "memory");
}

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
  if (kUseOnlyRepMovsb)
    return CopyRepMovsb(dst, src, count);

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
  if (kHasAvx && count < 256)
    return CopyBlockOverlap<128>(dst, src, count);
  if (count <= kRepMovsBSize)
    return CopyDstAlignedBlocks<kLoopCopyBlockSize, 32>(dst, src, count);
  return CopyRepMovsb(dst, src, count);
}

LLVM_LIBC_FUNCTION(void *, memcpy,
                   (void *__restrict dst, const void *__restrict src,
                    size_t size)) {
  memcpy_x86(reinterpret_cast<char *>(dst), reinterpret_cast<const char *>(src),
             size);
  return dst;
}

} // namespace __llvm_libc
