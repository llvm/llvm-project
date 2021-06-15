//===-- Implementation of memcpy ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memcpy.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/elements.h"

namespace __llvm_libc {

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
static void memcpy_impl(char *__restrict dst, const char *__restrict src,
                        size_t count) {
  // Use scalar strategies (_1, _2, _3 ...)
  using namespace __llvm_libc::scalar;

  if (count == 0)
    return;
  if (count == 1)
    return Copy<_1>(dst, src);
  if (count == 2)
    return Copy<_2>(dst, src);
  if (count == 3)
    return Copy<_3>(dst, src);
  if (count == 4)
    return Copy<_4>(dst, src);
  if (count < 8)
    return Copy<HeadTail<_4>>(dst, src, count);
  if (count < 16)
    return Copy<HeadTail<_8>>(dst, src, count);
  if (count < 32)
    return Copy<HeadTail<_16>>(dst, src, count);
  if (count < 64)
    return Copy<HeadTail<_32>>(dst, src, count);
  if (count < 128)
    return Copy<HeadTail<_64>>(dst, src, count);
  return Copy<Align<_32, Arg::Src>::Then<Loop<_32>>>(dst, src, count);
}

LLVM_LIBC_FUNCTION(void *, memcpy,
                   (void *__restrict dst, const void *__restrict src,
                    size_t size)) {
  memcpy_impl(reinterpret_cast<char *>(dst),
              reinterpret_cast<const char *>(src), size);
  return dst;
}

} // namespace __llvm_libc
