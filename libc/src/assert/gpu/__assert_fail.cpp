//===-- GPU definition of a libc internal assert macro ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/assert/__assert_fail.h"

#include "src/__support/CPP/atomic.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/libc_assert.h"
#include "src/stdlib/abort.h"

namespace __llvm_libc {

// A single-use lock to allow only a single thread to print the assertion.
static cpp::Atomic<uint32_t> lock = 0;

LLVM_LIBC_FUNCTION(void, __assert_fail,
                   (const char *assertion, const char *file, unsigned line,
                    const char *function)) {
  uint64_t mask = gpu::get_lane_mask();
  // We only want a single work group or warp to handle the assertion. Each
  // group attempts to claim the lock, if it is already claimed we simply exit.
  uint32_t claimed = gpu::is_first_lane(mask)
                         ? !lock.fetch_or(1, cpp::MemoryOrder::ACQUIRE)
                         : 0;
  if (!gpu::broadcast_value(mask, claimed)) {
#if defined(LIBC_TARGET_ARCH_IS_NVPTX)
    LIBC_INLINE_ASM("exit;" ::: "memory");
#elif defined(LIBC_TARGET_ARCH_IS_AMDGPU)
    __builtin_amdgcn_endpgm();
#endif
    __builtin_unreachable();
  }

  // Only a single line should be printed if an assertion is hit.
  if (gpu::is_first_lane(mask))
    __llvm_libc::report_assertion_failure(assertion, file, line, function);
  __llvm_libc::abort();
}

} // namespace __llvm_libc
