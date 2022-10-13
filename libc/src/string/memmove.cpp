//===-- Implementation of memmove -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memmove.h"

#include "src/__support/common.h"
#include "src/__support/integer_operations.h"
#include "src/string/memory_utils/elements.h"
#include <stddef.h> // size_t, ptrdiff_t

namespace __llvm_libc {

static inline void inline_memmove(char *dst, const char *src, size_t count) {
  using namespace __llvm_libc::scalar;
  if (count == 0)
    return;
  if (count == 1)
    return move<_1>(dst, src);
  if (count <= 4)
    return move<HeadTail<_2>>(dst, src, count);
  if (count <= 8)
    return move<HeadTail<_4>>(dst, src, count);
  if (count <= 16)
    return move<HeadTail<_8>>(dst, src, count);
  if (count <= 32)
    return move<HeadTail<_16>>(dst, src, count);
  if (count <= 64)
    return move<HeadTail<_32>>(dst, src, count);
  if (count <= 128)
    return move<HeadTail<_64>>(dst, src, count);

  using AlignedMoveLoop = Align<_16, Arg::Src>::Then<Loop<_64>>;
  if (dst < src)
    return move<AlignedMoveLoop>(dst, src, count);
  else if (dst > src)
    return move_backward<AlignedMoveLoop>(dst, src, count);
}

LLVM_LIBC_FUNCTION(void *, memmove,
                   (void *dst, const void *src, size_t count)) {
  inline_memmove(reinterpret_cast<char *>(dst),
                 reinterpret_cast<const char *>(src), count);
  return dst;
}

} // namespace __llvm_libc
