//===-- lib/cuda/allocatable.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/allocatable.h"
#include "flang-rt/runtime/assign-impl.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/stat.h"
#include "flang-rt/runtime/terminator.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/CUDA/descriptor.h"
#include "flang/Runtime/CUDA/memmove-function.h"
#include "flang/Runtime/allocatable.h"

#include "cuda.h"
#include "cuda_runtime.h"

namespace Fortran::runtime::cuda {

static bool deviceContextTornDown() {
  // Keep cudaGetLastError transparent: consume probe-only sticky errors when
  // the slot started clean, never discarding a pre-existing user error.
  cudaError_t priorErr{cudaPeekAtLastError()};
  // Prefer cleanup when state cannot be proven torn down (avoids leaks).
  bool tornDown{false};
  int device{0};
  if (cudaGetDevice(&device) == cudaSuccess) {
    // Driver API reports primary-context state without lazily creating one;
    // resolve via cudart to avoid a libcuda link (current device only).
    using GetStateFn = CUresult(CUDAAPI *)(CUdevice, unsigned *, int *);
    static GetStateFn getState{[]() -> GetStateFn {
      void *fn{nullptr};
      // Prefer ByVersion(driver): unversioned lookup uses the runtime version
      // and fails when the runtime is newer than the driver.
      int driverVersion{0};
      if (cudaDriverGetVersion(&driverVersion) == cudaSuccess &&
          cudaGetDriverEntryPointByVersion("cuDevicePrimaryCtxGetState", &fn,
              static_cast<unsigned>(driverVersion), cudaEnableDefault,
              nullptr) == cudaSuccess &&
          fn) {
        return reinterpret_cast<GetStateFn>(fn);
      }
      if (cudaGetDriverEntryPoint("cuDevicePrimaryCtxGetState", &fn,
              cudaEnableDefault, nullptr) == cudaSuccess &&
          fn) {
        return reinterpret_cast<GetStateFn>(fn);
      }
      return nullptr;
    }()};
    if (getState) {
      unsigned flags{0};
      int active{0};
      if (getState(device, &flags, &active) == CUDA_SUCCESS) {
        tornDown = active == 0;
        // A sticky error (e.g. an illegal kernel memory access) leaves the
        // primary context active but unusable: later calls all fail, so
        // scope-exit frees would abort an otherwise successful program. A
        // null free is a no-op that surfaces this without creating a context.
        if (!tornDown && cudaFree(nullptr) != cudaSuccess) {
          tornDown = true;
        }
      }
    }
  } else {
    tornDown = true;
  }
  if (priorErr == cudaSuccess && cudaPeekAtLastError() != cudaSuccess) {
    (void)cudaGetLastError();
  }
  return tornDown;
}

extern "C" {
RT_EXT_API_GROUP_BEGIN

int RTDEF(CUFAllocatableAllocateSync)(Descriptor &desc, int64_t *stream,
    bool *pinned, bool hasStat, const Descriptor *errMsg,
    const char *sourceFile, int sourceLine, bool deviceInit) {
  int stat{RTNAME(CUFAllocatableAllocate)(desc, stream, pinned, hasStat, errMsg,
      sourceFile, sourceLine, deviceInit)};
#ifndef RT_DEVICE_COMPILATION
  // Descriptor synchronization is only done when the allocation is done
  // from the host.
  if (stat == StatOk) {
    void *deviceAddr{
        RTNAME(CUFGetDeviceAddress)((void *)&desc, sourceFile, sourceLine)};
    RTNAME(CUFDescriptorSync)
    ((Descriptor *)deviceAddr, &desc, sourceFile, sourceLine);
  }
#endif
  return stat;
}

int RTDEF(CUFAllocatableAllocate)(Descriptor &desc, int64_t *stream,
    bool *pinned, bool hasStat, const Descriptor *errMsg,
    const char *sourceFile, int sourceLine, bool deviceInit) {
  // Perform the standard allocation.
  int stat{RTNAME(AllocatableAllocate)(desc, stream, hasStat, errMsg,
      sourceFile, sourceLine, deviceInit ? &MemcpyHostToDevice : nullptr)};
  if (pinned) {
    // Set pinned according to stat. More infrastructre is needed to set it
    // closer to the actual allocation call.
    *pinned = (stat == StatOk);
  }
  return stat;
}

int RTDEF(CUFAllocatableAllocateSource)(Descriptor &alloc,
    const Descriptor &source, int64_t *stream, bool *pinned, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine,
    bool sourceIsDevice) {
  int stat{RTNAME(CUFAllocatableAllocate)(
      alloc, stream, pinned, hasStat, errMsg, sourceFile, sourceLine)};
  if (stat == StatOk) {
    Terminator terminator{sourceFile, sourceLine};
    Fortran::runtime::DoFromSourceAssign(alloc, source, terminator,
        sourceIsDevice ? &MemmoveDeviceToHost : &MemmoveHostToDevice);
  }
  return stat;
}

int RTDEF(CUFAllocatableAllocateSourceSync)(Descriptor &alloc,
    const Descriptor &source, int64_t *stream, bool *pinned, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine,
    bool sourceIsDevice) {
  int stat;
  if (sourceIsDevice) {
    stat = RTNAME(CUFAllocatableAllocate)(
        alloc, stream, pinned, hasStat, errMsg, sourceFile, sourceLine);
  } else {
    stat = RTNAME(CUFAllocatableAllocateSync)(
        alloc, stream, pinned, hasStat, errMsg, sourceFile, sourceLine);
  }
  if (stat == StatOk) {
    Terminator terminator{sourceFile, sourceLine};
    Fortran::runtime::DoFromSourceAssign(alloc, source, terminator,
        sourceIsDevice ? &MemmoveDeviceToHost : &MemmoveHostToDevice);
  }
  return stat;
}

int RTDEF(CUFAllocatableDeallocate)(Descriptor &desc, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  // Perform the standard allocation.
  int stat{RTNAME(AllocatableDeallocate)(
      desc, hasStat, errMsg, sourceFile, sourceLine)};
#ifndef RT_DEVICE_COMPILATION
  // Descriptor synchronization is only done when the deallocation is done
  // from the host.
  if (stat == StatOk) {
    void *deviceAddr{
        RTNAME(CUFGetDeviceAddress)((void *)&desc, sourceFile, sourceLine)};
    RTNAME(CUFDescriptorSync)
    ((Descriptor *)deviceAddr, &desc, sourceFile, sourceLine);
  }
#endif
  return stat;
}

RT_EXT_API_GROUP_END

bool RTDEF(CUFDeviceIsActive)() { return !deviceContextTornDown(); }

} // extern "C"

} // namespace Fortran::runtime::cuda
