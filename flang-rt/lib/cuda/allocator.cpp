//===-- lib/cuda/allocator.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/allocator.h"
#include "flang-rt/runtime/allocator-registry.h"
#include "flang-rt/runtime/derived.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/environment.h"
#include "flang-rt/runtime/lock.h"
#include "flang-rt/runtime/stat.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/type-info.h"
#include "flang/Common/ISO_Fortran_binding_wrapper.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Support/Fortran.h"

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

struct DeviceAllocation {
  void *ptr;
  std::size_t size;
  cudaStream_t stream;
};

// Compare address values. nullptr will be sorted at the end of the array.
int compareDeviceAlloc(const void *a, const void *b) {
  const DeviceAllocation *deva = (const DeviceAllocation *)a;
  const DeviceAllocation *devb = (const DeviceAllocation *)b;
  if (deva->ptr == nullptr && devb->ptr == nullptr)
    return 0;
  if (deva->ptr == nullptr)
    return 1;
  if (devb->ptr == nullptr)
    return -1;
  return deva->ptr < devb->ptr ? -1 : (deva->ptr > devb->ptr ? 1 : 0);
}

// Dynamic array for tracking asynchronous allocations.
static DeviceAllocation *asyncDeviceAllocations = nullptr;
Lock asyncDeviceAllocationTableLock;
static int maxDeviceAllocations{512}; // Initial size
static int numDeviceAllocations{0};
static constexpr int allocNotFound{-1};

static void initAsyncDeviceAllocations() {
  if (!asyncDeviceAllocations) {
    asyncDeviceAllocations = static_cast<DeviceAllocation *>(
        malloc(maxDeviceAllocations * sizeof(DeviceAllocation)));
    if (!asyncDeviceAllocations) {
      Terminator terminator{__FILE__, __LINE__};
      terminator.Crash("Failed to allocate tracking array");
    }
  }
}

static void doubleAllocationArray() {
  unsigned newSize = maxDeviceAllocations * 2;
  DeviceAllocation *newArray = static_cast<DeviceAllocation *>(
      realloc(asyncDeviceAllocations, newSize * sizeof(DeviceAllocation)));
  if (!newArray) {
    Terminator terminator{__FILE__, __LINE__};
    terminator.Crash("Failed to reallocate tracking array");
  }
  asyncDeviceAllocations = newArray;
  maxDeviceAllocations = newSize;
}

int findAsyncDeviceAllocation(void *ptr) {
  if (numDeviceAllocations == 0) {
    return allocNotFound;
  }

  int left{0};
  int right{numDeviceAllocations - 1};

  if (left == right) {
    return left;
  }

  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (asyncDeviceAllocations[mid].ptr == ptr) {
      return mid;
    }
    if (asyncDeviceAllocations[mid].ptr < ptr) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return allocNotFound;
}

void insertAsyncDeviceAllocation(
    void *ptr, std::size_t size, cudaStream_t stream) {
  initAsyncDeviceAllocations();
  if (numDeviceAllocations >= maxDeviceAllocations) {
    doubleAllocationArray();
  }
  asyncDeviceAllocations[numDeviceAllocations].ptr = ptr;
  asyncDeviceAllocations[numDeviceAllocations].size = size;
  asyncDeviceAllocations[numDeviceAllocations].stream = stream;
  ++numDeviceAllocations;
  qsort(asyncDeviceAllocations, numDeviceAllocations, sizeof(DeviceAllocation),
      compareDeviceAlloc);
}

cudaStream_t getAsyncDeviceAllocationStream(int pos) {
  if (pos < 0 || pos >= numDeviceAllocations) {
    return nullptr;
  }
  return asyncDeviceAllocations[pos].stream;
}

void eraseAsyncDeviceAllocation(int pos) {
  if (pos < 0 || pos >= numDeviceAllocations) {
    return;
  }
  asyncDeviceAllocations[pos].ptr = nullptr;
  asyncDeviceAllocations[pos].size = 0;
  asyncDeviceAllocations[pos].stream = (cudaStream_t)0;
  qsort(asyncDeviceAllocations, numDeviceAllocations, sizeof(DeviceAllocation),
      compareDeviceAlloc);
  --numDeviceAllocations;
}

void CUFResetStream(cudaStream_t stream) {
  CriticalSection critical{asyncDeviceAllocationTableLock};
  for (int i = 0; i < numDeviceAllocations; ++i) {
    if (asyncDeviceAllocations[i].stream == stream) {
      asyncDeviceAllocations[i].stream = nullptr;
    }
  }
}

extern "C" {

void RTDEF(CUFRegisterAllocator)() {
  allocatorRegistry.Register(
      kPinnedAllocatorPos, {&CUFAllocPinned, CUFFreePinned});
  allocatorRegistry.Register(
      kDeviceAllocatorPos, {&CUFAllocDevice, CUFFreeDevice});
  allocatorRegistry.Register(
      kManagedAllocatorPos, {&CUFAllocManaged, CUFFreeManaged});
  allocatorRegistry.Register(
      kUnifiedAllocatorPos, {&CUFAllocUnified, CUFFreeUnified});
}

bool RTDEF(CUFDeviceIsActive)() { return !deviceContextTornDown(); }

cudaStream_t RTDECL(CUFGetAssociatedStream)(void *p) {
  int pos = findAsyncDeviceAllocation(p);
  if (pos >= 0) {
    cudaStream_t stream = asyncDeviceAllocations[pos].stream;
    return stream;
  }
  return nullptr;
}

int RTDECL(CUFSetAssociatedStream)(void *p, cudaStream_t stream) {
  if (p == nullptr) {
    return StatBaseNull;
  }
  int pos = findAsyncDeviceAllocation(p);
  if (pos >= 0) {
    asyncDeviceAllocations[pos].stream = stream;
  } else {
    CriticalSection critical{asyncDeviceAllocationTableLock};
    insertAsyncDeviceAllocation(p, 0, stream);
  }
  return StatOk;
}
}

void *CUFAllocPinned(std::size_t sizeInBytes,
    [[maybe_unused]] std::size_t alignment,
    [[maybe_unused]] std::int64_t *asyncObject) {
  void *p;
  CUDA_REPORT_IF_ERROR(cudaMallocHost((void **)&p, sizeInBytes));
  return p;
}

void CUFFreePinned(void *p) { cudaFreeHost(p); }

void *CUFAllocDevice(std::size_t sizeInBytes,
    [[maybe_unused]] std::size_t alignment, std::int64_t *asyncObject) {
  void *p;
  if (Fortran::runtime::executionEnvironment.cudaDeviceIsManaged) {
    CUDA_REPORT_IF_ERROR(
        cudaMallocManaged((void **)&p, sizeInBytes, cudaMemAttachGlobal));
  } else {
    if (asyncObject == nullptr) {
      CUDA_REPORT_IF_ERROR(cudaMalloc(&p, sizeInBytes));
    } else {
      CUDA_REPORT_IF_ERROR(
          cudaMallocAsync(&p, sizeInBytes, (cudaStream_t)*asyncObject));
      CriticalSection critical{asyncDeviceAllocationTableLock};
      insertAsyncDeviceAllocation(p, sizeInBytes, (cudaStream_t)*asyncObject);
    }
  }
  return p;
}

// Scope-exit cleanup is guarded in lowering; explicit deallocation after a
// reset is unsupported, keeping cudaFreeAsync free of context-query overhead.
void CUFFreeDevice(void *p) {
  CriticalSection critical{asyncDeviceAllocationTableLock};
  int pos = findAsyncDeviceAllocation(p);
  if (pos >= 0) {
    cudaStream_t stream = asyncDeviceAllocations[pos].stream;
    eraseAsyncDeviceAllocation(pos);
    CUDA_REPORT_IF_ERROR(cudaFreeAsync(p, stream));
  } else {
    CUDA_REPORT_IF_ERROR(cudaFree(p));
  }
}

void *CUFAllocManaged(std::size_t sizeInBytes,
    [[maybe_unused]] std::size_t alignment,
    [[maybe_unused]] std::int64_t *asyncObject) {
  void *p;
  CUDA_REPORT_IF_ERROR(
      cudaMallocManaged((void **)&p, sizeInBytes, cudaMemAttachGlobal));
  return reinterpret_cast<void *>(p);
}

void CUFFreeManaged(void *p) { CUDA_REPORT_IF_ERROR(cudaFree(p)); }

void *CUFAllocUnified(std::size_t sizeInBytes,
    [[maybe_unused]] std::size_t alignment,
    [[maybe_unused]] std::int64_t *asyncObject) {
  // Call alloc managed for the time being.
  return CUFAllocManaged(sizeInBytes, alignment, asyncObject);
}

void CUFFreeUnified(void *p) {
  // Call free managed for the time being.
  CUFFreeManaged(p);
}

} // namespace Fortran::runtime::cuda
