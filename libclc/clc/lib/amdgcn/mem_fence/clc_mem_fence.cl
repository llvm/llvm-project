//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/mem_fence/clc_mem_fence.h>

void __clc_amdgcn_s_waitcnt(unsigned flags);

// s_waitcnt takes 16bit argument with a combined number of maximum allowed
// pending operations:
// [12:8] LGKM -- LDS, GDS, Konstant (SMRD), Messages
// [7] -- undefined
// [6:4] -- exports, GDS, and mem write
// [3:0] -- vector memory operations

// Newer clang supports __builtin_amdgcn_s_waitcnt
#if __clang_major__ >= 5
#define __waitcnt(x) __builtin_amdgcn_s_waitcnt(x)
#else
#define __waitcnt(x) __clc_amdgcn_s_waitcnt(x)
_CLC_DEF void __clc_amdgcn_s_waitcnt(unsigned) __asm("llvm.amdgcn.s.waitcnt");
#endif

_CLC_OVERLOAD _CLC_DEF void __clc_mem_fence(Scope scope,
                                            MemorySemantics semantics) {
  if (semantics & CrossWorkgroupMemory) {
    // scalar loads are counted with LGKM but we don't know whether
    // the compiler turned any loads to scalar
    __waitcnt(0);
  } else if (semantics & WorkgroupMemory)
    __waitcnt(0xff); // LGKM is [12:8]
}
#undef __waitcnt
