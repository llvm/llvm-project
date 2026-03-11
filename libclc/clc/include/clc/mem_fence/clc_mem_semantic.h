//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLC_MEM_FENCE_CLC_MEM_SEMANTIC_H
#define CLC_MEM_FENCE_CLC_MEM_SEMANTIC_H

// The memory or address space to which the memory ordering is applied.
typedef enum __CLC_MemorySemantics {
  __CLC_MEMORY_PRIVATE = 1 << 0,
  __CLC_MEMORY_GLOBAL = 1 << 1,
  __CLC_MEMORY_CONSTANT = 1 << 2,
  __CLC_MEMORY_LOCAL = 1 << 3,
  __CLC_MEMORY_GENERIC = 1 << 4,
} __CLC_MemorySemantics;

#endif // CLC_MEM_FENCE_CLC_MEM_SEMANTIC_H
