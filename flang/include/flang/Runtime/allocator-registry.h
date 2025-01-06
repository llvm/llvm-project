//===-- runtime/allocator-registry.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_ALLOCATOR_REGISTRY_H_
#define FORTRAN_RUNTIME_ALLOCATOR_REGISTRY_H_

#include "flang/Common/api-attrs.h"
#include "flang/Runtime/allocator-registry-consts.h"
#include <cstdlib>
#include <vector>

#define MAX_ALLOCATOR 7 // 3 bits are reserved in the descriptor.

namespace Fortran::runtime {

using AllocFct = void *(*)(std::size_t);
using FreeFct = void (*)(void *);

typedef struct Allocator_t {
  AllocFct alloc{nullptr};
  FreeFct free{nullptr};
} Allocator_t;

#ifdef RT_DEVICE_COMPILATION
static RT_API_ATTRS void *MallocWrapper(std::size_t size) {
  return std::malloc(size);
}
static RT_API_ATTRS void FreeWrapper(void *p) { return std::free(p); }
#endif

struct AllocatorRegistry {
#ifdef RT_DEVICE_COMPILATION
  RT_API_ATTRS constexpr AllocatorRegistry()
      : allocators{{&MallocWrapper, &FreeWrapper}} {}
#else
  constexpr AllocatorRegistry() {
    allocators[kDefaultAllocator] = {&std::malloc, &std::free};
  };
#endif
  RT_API_ATTRS void Register(int, Allocator_t);
  RT_API_ATTRS AllocFct GetAllocator(int pos);
  RT_API_ATTRS FreeFct GetDeallocator(int pos);

  Allocator_t allocators[MAX_ALLOCATOR];
};

RT_OFFLOAD_VAR_GROUP_BEGIN
extern RT_VAR_ATTRS AllocatorRegistry allocatorRegistry;
RT_OFFLOAD_VAR_GROUP_END

} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_ALLOCATOR_REGISTRY_H_
