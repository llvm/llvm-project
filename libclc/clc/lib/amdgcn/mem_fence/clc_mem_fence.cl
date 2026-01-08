//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/mem_fence/clc_mem_fence.h>

#define BUILTIN_FENCE_ORDER(memory_order, ...)                                 \
  switch (memory_order) {                                                      \
  case __ATOMIC_ACQUIRE:                                                       \
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, __VA_ARGS__);                     \
    break;                                                                     \
  case __ATOMIC_RELEASE:                                                       \
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, __VA_ARGS__);                     \
    break;                                                                     \
  case __ATOMIC_ACQ_REL:                                                       \
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, __VA_ARGS__);                     \
    break;                                                                     \
  case __ATOMIC_SEQ_CST:                                                       \
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, __VA_ARGS__);                     \
    break;                                                                     \
  default:                                                                     \
    __builtin_unreachable();                                                   \
  }                                                                            \
  break;

#define BUILTIN_FENCE(memory_scope, memory_order, ...)                         \
  switch (memory_scope) {                                                      \
  case __MEMORY_SCOPE_DEVICE:                                                  \
    BUILTIN_FENCE_ORDER(memory_order, "agent", ##__VA_ARGS__)                  \
  case __MEMORY_SCOPE_WRKGRP:                                                  \
    BUILTIN_FENCE_ORDER(memory_order, "workgroup", ##__VA_ARGS__)              \
  case __MEMORY_SCOPE_WVFRNT:                                                  \
    BUILTIN_FENCE_ORDER(memory_order, "wavefront", ##__VA_ARGS__)              \
  case __MEMORY_SCOPE_SINGLE:                                                  \
    BUILTIN_FENCE_ORDER(memory_order, "singlethread", ##__VA_ARGS__)           \
  case __MEMORY_SCOPE_SYSTEM:                                                  \
  default:                                                                     \
    BUILTIN_FENCE_ORDER(memory_order, "", ##__VA_ARGS__)                       \
  }

_CLC_OVERLOAD _CLC_DEF void
__clc_mem_fence(int memory_scope, int memory_order,
                __CLC_MemorySemantics memory_semantics) {
  if (memory_semantics == __CLC_MEMORY_LOCAL) {
    BUILTIN_FENCE(memory_scope, memory_order, "local")
  } else if (memory_semantics == __CLC_MEMORY_GLOBAL) {
    BUILTIN_FENCE(memory_scope, memory_order, "global")
  } else if (memory_semantics == (__CLC_MEMORY_LOCAL | __CLC_MEMORY_GLOBAL)) {
    BUILTIN_FENCE(memory_scope, memory_order, "local", "global")
  } else {
    BUILTIN_FENCE(memory_scope, memory_order)
  }
}
