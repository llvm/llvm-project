//===-- Implementation of memmove -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memmove.h"

#include "src/__support/common.h"
#include "src/string/memory_utils/op_aarch64.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_x86.h"
#include <stddef.h> // size_t, ptrdiff_t

#include <stdio.h>

namespace __llvm_libc {

static inline void inline_memmove(char *dst, const char *src, size_t count) {
#if defined(LLVM_LIBC_ARCH_X86)
  static constexpr size_t kMaxSize = x86::kAvx512F ? 64
                                     : x86::kAvx   ? 32
                                     : x86::kSse2  ? 16
                                                   : 8;
#elif defined(LLVM_LIBC_ARCH_AARCH64)
  static constexpr size_t kMaxSize = aarch64::kNeon ? 16 : 8;
#else
  static constexpr size_t kMaxSize = 8;
#endif
  if (count == 0)
    return;
  if (count == 1)
    return generic::Memmove<1, kMaxSize>::block(dst, src);
  if (count <= 4)
    return generic::Memmove<2, kMaxSize>::head_tail(dst, src, count);
  if (count <= 8)
    return generic::Memmove<4, kMaxSize>::head_tail(dst, src, count);
  if (count <= 16)
    return generic::Memmove<8, kMaxSize>::head_tail(dst, src, count);
  if (count <= 32)
    return generic::Memmove<16, kMaxSize>::head_tail(dst, src, count);
  if (count <= 64)
    return generic::Memmove<32, kMaxSize>::head_tail(dst, src, count);
  if (count <= 128)
    return generic::Memmove<64, kMaxSize>::head_tail(dst, src, count);

  if (dst < src) {
    generic::Memmove<32, kMaxSize>::align_forward<Arg::Src>(dst, src, count);
    return generic::Memmove<64, kMaxSize>::loop_and_tail_forward(dst, src,
                                                                 count);
  } else {
    generic::Memmove<32, kMaxSize>::align_backward<Arg::Src>(dst, src, count);
    return generic::Memmove<64, kMaxSize>::loop_and_tail_backward(dst, src,
                                                                  count);
  }
}

LLVM_LIBC_FUNCTION(void *, memmove,
                   (void *dst, const void *src, size_t count)) {
  inline_memmove(reinterpret_cast<char *>(dst),
                 reinterpret_cast<const char *>(src), count);
  return dst;
}

} // namespace __llvm_libc
