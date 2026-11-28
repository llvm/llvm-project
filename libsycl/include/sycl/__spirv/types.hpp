//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains common SPIR-V enum definitions used by libsycl.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___SPIRV_TYPES_HPP
#define _LIBSYCL___SPIRV_TYPES_HPP

#include <cstdint>

namespace __spirv {

enum Scope : std::int32_t {
  CrossDevice = 0,
  Device = 1,
  Workgroup = 2,
  Subgroup = 3,
  Invocation = 4,
};

enum MemorySemanticsMask : std::int32_t {
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
};

// SPIRV 3.2.6. Storage Class
// Specification declares more storage classes, but only a SYCL-used subset is
// included here.
enum class StorageClass : std::int32_t {
  Workgroup = 4,
  CrossWorkgroup = 5,
  Function = 7
};

} // namespace __spirv

#endif // _LIBSYCL___SPIRV_TYPES_HPP
