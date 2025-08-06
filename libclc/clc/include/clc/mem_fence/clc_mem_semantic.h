//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_MEM_FENCE_CLC_MEM_SEMANTIC_H__
#define __CLC_MEM_FENCE_CLC_MEM_SEMANTIC_H__

// The memory or address space to which the memory ordering is applied.
typedef enum MemorySemantic {
  MEMORY_PRIVATE = 0x1,
  MEMORY_LOCAL = 0x2,
  MEMORY_GLOBAL = 0x4,
  MEMORY_CONSTANT = 0x8,
  MEMORY_GENERIC = 0x10
} MemorySemantic;

#endif // __CLC_MEM_FENCE_CLC_MEM_SEMANTIC_H__
