//===-- scudo_crc32.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// CRC32 function leveraging hardware specific instructions. This has to be
/// kept separated to restrict the use of compiler specific flags to this file.
///
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_internal_defs.h"

// Hardware CRC32 is supported at compilation via the following:
// - for i386 & x86_64: -msse4.2
// - for ARM & AArch64: -march=armv8-a+crc or -mcrc
// An additional check must be performed at runtime as well to make sure the
// emitted instructions are valid on the target host.

#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
# ifdef __SSE4_2__
#  include <smmintrin.h>
#  define CRC32_INTRINSIC FIRST_32_SECOND_64(_mm_crc32_u32, _mm_crc32_u64)
# endif
# ifdef __ARM_FEATURE_CRC32
#  include <arm_acle.h>
#  define CRC32_INTRINSIC FIRST_32_SECOND_64(__crc32cw, __crc32cd)
# endif
#endif  // defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)

namespace __scudo {

#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
u32 computeHardwareCRC32(u32 Crc, uptr Data) {
  return CRC32_INTRINSIC(Crc, Data);
}
#endif  // defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)

}  // namespace __scudo
