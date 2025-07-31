//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_MEM_FENCE_CLC_MEM_SCOPE_SEMANTICS_H__
#define __CLC_MEM_FENCE_CLC_MEM_SCOPE_SEMANTICS_H__

// Scope values are defined in SPIR-V spec.
typedef enum Scope {
  CrossDevice = 0,
  Device = 1,
  Workgroup = 2,
  Subgroup = 3,
  Invocation = 4,
} Scope;

// MemorySemantics values are defined in SPIR-V spec.
typedef enum MemorySemantics {
  None = 0x0,
  Acquire = 0x2,
  Release = 0x4,
  AcquireRelease = 0x8,
  SequentiallyConsistent = 0x10,
  UniformMemory = 0x40,
  SubgroupMemory = 0x80,
  WorkgroupMemory = 0x100,
  CrossWorkgroupMemory = 0x200,
  AtomicCounterMemory = 0x400,
  ImageMemory = 0x800,
} MemorySemantics;

#endif // __CLC_MEM_FENCE_CLC_MEM_SCOPE_SEMANTICS_H__
