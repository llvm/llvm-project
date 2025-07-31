//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/synchronization/barrier.h>
#include <clc/opencl/synchronization/utils.h>
#include <clc/synchronization/clc_barrier.h>

_CLC_DEF _CLC_OVERLOAD void barrier(cl_mem_fence_flags flags) {
  Scope scope = getCLCScope(memory_scope_device);
  MemorySemantics semantics = getCLCMemorySemantics(flags);
  __clc_barrier(scope, semantics);
}
