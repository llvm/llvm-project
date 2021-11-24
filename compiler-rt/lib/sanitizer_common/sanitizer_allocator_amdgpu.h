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
  static bool Init();
  static void *Allocate(uptr size, uptr alignment,
                        DeviceAllocationInfo *da_info);
  static void Deallocate(void *p);
  static bool GetPointerInfo(uptr ptr, DevivePointerInfo *ptr_info);
  static uptr GetPageSize();
};

struct AmdgpuAllocationInfo : public DeviceAllocationInfo {
  AmdgpuAllocationInfo() : DeviceAllocationInfo(DAT_AMDGPU) {
    status = HSA_STATUS_SUCCESS;
    alloc_func = nullptr;
  }
  hsa_status_t status;
  void *alloc_func;
  hsa_amd_memory_pool_t memory_pool;
  size_t size;
  uint32_t flags;
  void *ptr;
};
#endif  // SANITIZER_AMDGPU
