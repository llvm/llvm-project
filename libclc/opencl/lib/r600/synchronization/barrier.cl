//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/explicit_fence/explicit_memory_fence.h>

_CLC_DEF void __clc_r600_barrier(void) __asm("llvm.r600.group.barrier");

_CLC_DEF _CLC_OVERLOAD void barrier(uint flags) {
  // We should call mem_fence here, but that is not implemented for r600 yet
  __clc_r600_barrier();
}
