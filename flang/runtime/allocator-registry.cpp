//===-- runtime/allocator-registry.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/allocator-registry.h"
#include "terminator.h"

namespace Fortran::runtime {

#ifndef FLANG_RUNTIME_NO_GLOBAL_VAR_DEFS
RT_OFFLOAD_VAR_GROUP_BEGIN
RT_VAR_ATTRS AllocatorRegistry allocatorRegistry;
RT_OFFLOAD_VAR_GROUP_END
#endif // FLANG_RUNTIME_NO_GLOBAL_VAR_DEFS

RT_OFFLOAD_API_GROUP_BEGIN
RT_API_ATTRS void AllocatorRegistry::Register(int pos, Allocator_t allocator) {
  // pos 0 is reserved for the default allocator and is registered in the
  // struct ctor.
  INTERNAL_CHECK(pos > 0 && pos < MAX_ALLOCATOR);
  allocators[pos] = allocator;
}

RT_API_ATTRS AllocFct AllocatorRegistry::GetAllocator(int pos) {
  INTERNAL_CHECK(pos >= 0 && pos < MAX_ALLOCATOR);
  AllocFct f{allocators[pos].alloc};
  INTERNAL_CHECK(f != nullptr);
  return f;
}

RT_API_ATTRS FreeFct AllocatorRegistry::GetDeallocator(int pos) {
  INTERNAL_CHECK(pos >= 0 && pos < MAX_ALLOCATOR);
  FreeFct f{allocators[pos].free};
  INTERNAL_CHECK(f != nullptr);
  return f;
}
RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime
