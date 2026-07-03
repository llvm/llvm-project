//===-- include/flang-rt/runtime/allocator-registry.h -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_RT_RUNTIME_ALLOCATOR_REGISTRY_H_
#define FLANG_RT_RUNTIME_ALLOCATOR_REGISTRY_H_

#include "flang/Common/api-attrs.h"
#include "flang/Runtime/allocator-registry-consts.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define MAX_ALLOCATOR 7 // 3 bits are reserved in the descriptor.

namespace Fortran::runtime {

using AllocFct = void *(*)(std::size_t, std::size_t, std::int64_t *);
using FreeFct = void (*)(void *);

typedef struct Allocator_t {
  AllocFct alloc{nullptr};
  FreeFct free{nullptr};
} Allocator_t;

static RT_API_ATTRS void *MallocWrapper(std::size_t size,
    [[maybe_unused]] std::size_t alignment, [[maybe_unused]] std::int64_t *) {
#if !defined(RT_DEVICE_COMPILATION) && !defined(_WIN32)
  // std::malloc only guarantees alignof(std::max_align_t). When a larger
  // alignment is requested, an aligned allocation routine must be used.
  if (alignment > alignof(std::max_align_t)) {
#if defined(__APPLE__)
    // std::aligned_alloc is only available on macOS 10.15 and newer, but
    // flang-rt is built with an older deployment target. posix_memalign is
    // available on all supported macOS versions.
    void *ptr{nullptr};
    if (posix_memalign(&ptr, alignment, size) != 0) {
      return nullptr;
    }
    return ptr;
#else
    // Round size up to a multiple of the alignment, as required by
    // std::aligned_alloc.
    if (size > SIZE_MAX - alignment) {
      return nullptr;
    }
    std::size_t alignedSize{((size + alignment - 1) / alignment) * alignment};
    return std::aligned_alloc(alignment, alignedSize);
#endif
  }
#endif
  return std::malloc(size);
}
#ifdef RT_DEVICE_COMPILATION
static RT_API_ATTRS void FreeWrapper(void *p) { return std::free(p); }
#endif

struct AllocatorRegistry {
#ifdef RT_DEVICE_COMPILATION
  RT_API_ATTRS constexpr AllocatorRegistry()
      : allocators{{&MallocWrapper, &FreeWrapper}} {}
#else
  constexpr AllocatorRegistry() {
    allocators[kDefaultAllocator] = {&MallocWrapper, &std::free};
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

#endif // FLANG_RT_RUNTIME_ALLOCATOR_REGISTRY_H_
