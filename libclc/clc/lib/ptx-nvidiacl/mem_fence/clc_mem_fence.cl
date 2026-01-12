//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/mem_fence/clc_mem_fence.h>

_CLC_OVERLOAD _CLC_DEF void
__clc_mem_fence(int memory_scope, int memory_order,
                __CLC_MemorySemantics memory_semantics) {
  if (memory_scope & (__MEMORY_SCOPE_DEVICE | __MEMORY_SCOPE_WRKGRP))
    __nvvm_membar_cta();
}
