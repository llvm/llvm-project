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

#include "cuda_runtime.h"

namespace Fortran::runtime::cuda {

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
static DeviceAllocation *deviceAllocations = nullptr;
Lock lock;
static int maxDeviceAllocations{512}; // Initial size
static int numDeviceAllocations{0};
static constexpr int allocNotFound{-1};

static void initAllocations() {
  if (!deviceAllocations) {
    deviceAllocations = static_cast<DeviceAllocation *>(
        malloc(maxDeviceAllocations * sizeof(DeviceAllocation)));
    if (!deviceAllocations) {
      Terminator terminator{__FILE__, __LINE__};
      terminator.Crash("Failed to allocate tracking array");
    }
  }
}

static void doubleAllocationArray() {
  unsigned newSize = maxDeviceAllocations * 2;
  DeviceAllocation *newArray = static_cast<DeviceAllocation *>(
      realloc(deviceAllocations, newSize * sizeof(DeviceAllocation)));
  if (!newArray) {
    Terminator terminator{__FILE__, __LINE__};
    terminator.Crash("Failed to reallocate tracking array");
  }
  deviceAllocations = newArray;
  maxDeviceAllocations = newSize;
}

static unsigned findAllocation(void *ptr) {
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
    if (deviceAllocations[mid].ptr == ptr) {
      return mid;
    }
    if (deviceAllocations[mid].ptr < ptr) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return allocNotFound;
}

static void insertAllocation(void *ptr, std::size_t size, std::int64_t stream) {
  CriticalSection critical{lock};
  initAllocations();
  if (numDeviceAllocations >= maxDeviceAllocations) {
    doubleAllocationArray();
  }
  deviceAllocations[numDeviceAllocations].ptr = ptr;
  deviceAllocations[numDeviceAllocations].size = size;
  deviceAllocations[numDeviceAllocations].stream = (cudaStream_t)stream;
  ++numDeviceAllocations;
  qsort(deviceAllocations, numDeviceAllocations, sizeof(DeviceAllocation),
      compareDeviceAlloc);
}

static void eraseAllocation(int pos) {
  deviceAllocations[pos].ptr = nullptr;
  deviceAllocations[pos].size = 0;
  deviceAllocations[pos].stream = (cudaStream_t)0;
  qsort(deviceAllocations, numDeviceAllocations, sizeof(DeviceAllocation),
      compareDeviceAlloc);
  --numDeviceAllocations;
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
}

void *CUFAllocPinned(
    std::size_t sizeInBytes, [[maybe_unused]] std::int64_t *asyncObject) {
  void *p;
  CUDA_REPORT_IF_ERROR(cudaMallocHost((void **)&p, sizeInBytes));
  return p;
}

void CUFFreePinned(void *p) { CUDA_REPORT_IF_ERROR(cudaFreeHost(p)); }

void *CUFAllocDevice(std::size_t sizeInBytes, std::int64_t *asyncObject) {
  void *p;
  if (Fortran::runtime::executionEnvironment.cudaDeviceIsManaged) {
    CUDA_REPORT_IF_ERROR(
        cudaMallocManaged((void **)&p, sizeInBytes, cudaMemAttachGlobal));
  } else {
    if (asyncObject == kNoAsyncObject) {
      CUDA_REPORT_IF_ERROR(cudaMalloc(&p, sizeInBytes));
    } else {
      CUDA_REPORT_IF_ERROR(
          cudaMallocAsync(&p, sizeInBytes, (cudaStream_t)*asyncObject));
      insertAllocation(p, sizeInBytes, (cudaStream_t)*asyncObject);
    }
  }
  return p;
}

void CUFFreeDevice(void *p) {
  CriticalSection critical{lock};
  int pos = findAllocation(p);
  if (pos >= 0) {
    cudaStream_t stream = deviceAllocations[pos].stream;
    eraseAllocation(pos);
    CUDA_REPORT_IF_ERROR(cudaFreeAsync(p, stream));
  } else {
    CUDA_REPORT_IF_ERROR(cudaFree(p));
  }
}

void *CUFAllocManaged(
    std::size_t sizeInBytes, [[maybe_unused]] std::int64_t *asyncObject) {
  void *p;
  CUDA_REPORT_IF_ERROR(
      cudaMallocManaged((void **)&p, sizeInBytes, cudaMemAttachGlobal));
  return reinterpret_cast<void *>(p);
}

void CUFFreeManaged(void *p) { CUDA_REPORT_IF_ERROR(cudaFree(p)); }

void *CUFAllocUnified(
    std::size_t sizeInBytes, [[maybe_unused]] std::int64_t *asyncObject) {
  // Call alloc managed for the time being.
  return CUFAllocManaged(sizeInBytes, asyncObject);
}

void CUFFreeUnified(void *p) {
  // Call free managed for the time being.
  CUFFreeManaged(p);
}

} // namespace Fortran::runtime::cuda
