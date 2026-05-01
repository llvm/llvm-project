//===-- lib/openmp/omp_alloc.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define ALLOC_DEBUG 1

#include "flang/Runtime/OpenMP/omp_alloc.h"
#include "flang-rt/runtime/allocator-registry.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/terminator.h"
#include "flang/Runtime/OpenMP/omp_util.h"
#include "flang/Support/Fortran.h"
#include <cstdio>
#include <cstdlib>

namespace Fortran::runtime::omp {

static bool debugEnabled;

// Declare OpenMP memory management routines to avoid importing
// definitions via "omp.h" (and thus create a dependency to the
// OpenMP runtime library code).
extern "C" int omp_get_default_device(void);
extern "C" void *omp_target_alloc(std::size_t, int);
extern "C" void omp_target_free(void *, int);

// Track which device each pointer was allocated on so that
// OpenMPFree can pass the correct device ID to omp_target_free,
// even if omp_set_default_device() was called between ALLOCATE
// and DEALLOCATE.
static PointerDeviceMap allocDeviceMap;

/// Allocate \p AllocationSize bytes on the current default OpenMP device.
static void *OpenMPAlloc(std::size_t AllocationSize, std::int64_t *) {
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr, "[OMP_ALLOC] %s(%zu) (%s:%d)\n", __PRETTY_FUNCTION__,
        AllocationSize, __FILE__, __LINE__);
  }
#endif
  int device{omp_get_default_device()};
  void *pointer{omp_target_alloc(AllocationSize, device)};
  if (pointer) {
    allocDeviceMap.insert(pointer, device);
  }
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr,
        "[OMP_ALLOC] pointer of size %zu allocated at %p"
        " on device %d.\n",
        AllocationSize, pointer, device);
  }
#endif
  return pointer;
}

/// Free a pointer previously allocated by OpenMPAlloc on the correct device.
static void OpenMPFree(void *pointer) {
  int device{allocDeviceMap.removeAndGet(pointer)};
  if (device == -1) {
    Terminator{__FILE__, __LINE__}.Crash(
        "OpenMPFree: pointer %p was not allocated by OpenMPAlloc", pointer);
  }
#if ALLOC_DEBUG
  if (debugEnabled) {
    std::fprintf(stderr, "[OMP_ALLOC] %s(%p) device %d (%s:%d)\n",
        __PRETTY_FUNCTION__, pointer, device, __FILE__, __LINE__);
  }
#endif
  omp_target_free(pointer, device);
}

extern "C" {
void RTDEF(OpenMPRegisterAllocator)() {
#if ALLOC_DEBUG
  debugEnabled = false;
  if (const char *env = std::getenv("OMP_ALLOC_DEBUG")) {
    debugEnabled = env[0] != '0' && env[0] != '\0';
  }
  if (debugEnabled) {
    std::fprintf(stderr, "[OMP_ALLOC] %s (%s:%d)\n", __PRETTY_FUNCTION__,
        __FILE__, __LINE__);
    std::fprintf(
        stderr, "[OMP_ALLOC] registering OpenMP device memory allocator\n");
  }
#endif
  allocatorRegistry.Register(1, {&OpenMPAlloc, &OpenMPFree});
}

void RTDEF(OpenMPAllocatableSetAllocIdx)(Descriptor &descriptor, int pos) {
  if (descriptor.IsAllocatable() && !descriptor.IsAllocated()) {
#if ALLOC_DEBUG
    if (debugEnabled) {
      std::fprintf(
          stderr, "[OMP_ALLOC] OpenMPAllocatableSetAllocIdx = %d \n", pos);
    }
#endif
    descriptor.SetAllocIdx(pos);
  }
}
} // extern "C"

} // namespace Fortran::runtime::omp