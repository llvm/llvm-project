//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/synchronization/barrier.h>

_CLC_DEF _CLC_OVERLOAD void barrier(cl_mem_fence_flags flags) {
  __syncthreads();
}
