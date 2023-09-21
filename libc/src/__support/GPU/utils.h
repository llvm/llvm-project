//===---------------- Implementation of GPU utils ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_GPU_UTIL_H
#define LLVM_LIBC_SRC___SUPPORT_GPU_UTIL_H

#include "src/__support/macros/properties/architectures.h"

#if defined(LIBC_TARGET_ARCH_IS_AMDGPU)
#include "amdgpu/utils.h"
#elif defined(LIBC_TARGET_ARCH_IS_NVPTX)
#include "nvptx/utils.h"
#else
#include "generic/utils.h"
#endif

namespace __llvm_libc {
namespace gpu {
/// Get the first active thread inside the lane.
LIBC_INLINE uint64_t get_first_lane_id(uint64_t lane_mask) {
  return __builtin_ffsl(lane_mask) - 1;
}

/// Conditional that is only true for a single thread in a lane.
LIBC_INLINE bool is_first_lane(uint64_t lane_mask) {
  return gpu::get_lane_id() == get_first_lane_id(lane_mask);
}

} // namespace gpu
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_IO_H
