//===-- sanitizer_allocator_amdgpu.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the Sanitizer Allocator.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_ALLOCATOR_H
#  error This file must be included inside sanitizer_allocator_device.h
#endif

#if SANITIZER_AMDGPU
class AmdgpuMemFuncs {
 public:
  enum DeviceAllocFailure {
    kNotInitialized,
    kOutOfResources,
  };

  static bool Init();
  static void* Allocate(uptr size, uptr alignment,
                        DeviceAllocationInfo* da_info);
  static void Deallocate(void* p);
  static bool GetPointerInfo(uptr ptr, DevicePointerInfo* ptr_info);
  static uptr GetPageSize();
  static void RegisterSystemEventHandlers();
  static bool IsAmdgpuRuntimeShutdown();
  static void ClearAmdgpuRuntimeShutdownState();
  // Record an HSA error on da_info when DeviceAllocatorT fails before/without a
  // successful AmdgpuMemFuncs::Allocate (keeps HSA types out of device.h).
  static void NoteDeviceAllocatorFailure(DeviceAllocationInfo* da_info,
                                         DeviceAllocFailure failure);

 private:
  static void NotifyAmdgpuRuntimeShutdown();
};

struct AmdgpuAllocationInfo : public DeviceAllocationInfo {
  AmdgpuAllocationInfo() : DeviceAllocationInfo(DAT_AMDGPU) {
    status = HSA_STATUS_SUCCESS;
    alloc_func = nullptr;
  }
  // If allocation fails without an HSA API status, record one so callers never
  // see *ptr == nullptr with status still SUCCESS.
  void EnsureFailureStatus(hsa_status_t err) {
    if (status == HSA_STATUS_SUCCESS)
      status = err;
  }
  hsa_status_t status;
  void* alloc_func;
  hsa_amd_memory_pool_t memory_pool;
  u64 alignment;
  u64 address;
  u64 flags64;
  usize size;
  u32 flags;
  void* ptr;
};
#endif  // SANITIZER_AMDGPU
